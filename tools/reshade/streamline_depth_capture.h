// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "scene_depth_source.h"
#include <cstddef>
#include <cstdint>
#include <memory>

namespace sunshine_streamline::depth_capture {
  // An independent strong reference obtained while the application's API tag is
  // valid. This preserves the COM object, never the contents of its texture.
  struct source_reference;
  using source_ref = std::shared_ptr<const source_reference>;
  struct texture_reference;

  struct input : sunshine_scene_depth::frame {
    source_ref source;
    std::uint64_t source_present_generation{}; // Frozen separately at each tag entry.
    // Volatile API tags must copy at this call, even for a tracked ReShade DSV.
    bool force_snapshot{};
  };
  struct packet {
    input metadata;
    std::shared_ptr<const texture_reference> ownership;
    // queue is the requested consumer queue; the producer remains explicit
    // when an immutable copy is handed to a different queue on the same device.
    std::uint64_t texture{}, shader_resource{}, device{}, queue{}, producer_queue{}, capture_id{}, producer_fence{};
    std::uint64_t resource_id{};
    // This packet nominates a source only. The ReShade preservation owner must
    // separately prove and provide its current copy; texture/view stay zero.
    bool shared_preservation{};
    // A successful source nomination can own this presentation while capture
    // is unsupported/pending. This authorizes neither texture access nor a
    // fallback selector. Shared preservation proves pixel readiness separately.
    bool pixel_ready{};
    // Allocation dimensions stay intact (D3D12 depth/stencil copies require a
    // full subresource). Area is always positive and in allocation pixels;
    // absent tag extents normalize to the full allocation. Consumers map this
    // rectangle consistently for depth statistics and shader sampling.
    std::uint32_t width{}, height{}, format{}, srv_format{};
    sunshine_scene_depth::extent area;
  };
  // The adapter supplies the observed FG mode outside the capture lock. When
  // required, missing SL FG input remains unavailable instead of selecting a
  // different provider. This policy changes selection, never GPU ownership.
  struct selection_policy {
    bool require_frame_generation{};
    std::uint64_t epoch{};
    std::uint32_t viewport{};
  };
  enum class status {
    inactive, unavailable, malformed, unsupported_state, missing_state, conflicting_state, incomplete_state, unsupported_lifetime,
    unsupported_resource, unsupported_queue, exhausted, recorded, submitted,
    ready, stale, ambiguous, failed
  };
  enum class capture_failure {
    none, observer_loss, discarded_recording, close_failed, replay,
    producer_signal_failed, producer_queue_changed, consumer_signal_failed,
    consumer_queue_changed, evaluation_failed,
    evaluation_observation_changed, queue_retired, consumer_capacity, source_retired
  };
  // Selection evidence only; never grants GPU access to a pending snapshot.
  enum class selection_reason {
    not_attempted, no_current_nomination, inactive_views, ambiguous_views,
    pending_nomination, current_capture, current_already_consumed, no_admissible_capture,
    no_completed_snapshot, completed_before_gap, completed_not_older,
    completed_already_consumed, presentation_already_selected, layout_changed,
    completed_not_readable, completed_snapshot, fg_scope_missing, fg_scope_mismatch
  };
  // Acquisition authority, frozen under the capture lock. Diagnostics may be
  // omitted or reformatted without changing these source/continuity decisions.
  // This never authorizes reading pending pixels; packet::pixel_ready and the
  // normal consumer lease still govern every GPU read.
  struct acquisition_decision {
    bool source_selected{};
    sunshine_scene_depth::provider_kind provider{sunshine_scene_depth::provider_kind::streamline};
    std::uint64_t epoch{}, sequence{}, source_id{}, capture_id{};
    std::uint32_t viewport{};
    bool source_valid{}, repeated_frame{}, pending_frame{};
    // Capture-side proof for the provider's existing one-presentation NGX hold.
    // Nonzero identifies the exact consumed predecessor; the provider must
    // still match its private display copy, metadata, shape, age and queue.
    std::uint64_t pending_ngx_previous_capture{};
  };
  // Observation only. Rendering and source reuse consume acquisition_decision.
  struct capture_diagnostic {
    status result{status::unavailable};
    capture_failure failure{capture_failure::none};
    selection_reason selection{selection_reason::not_attempted};
    // Set only when the newest eligible completed snapshot is exactly the
    // already-consumed sequence. A provider may hold its private display copy;
    // this does not permit reading the newer pending texture.
    std::uint64_t consumed_completed_capture{};
    std::uint64_t capture_id{}, sequence{}, epoch{}, command{}, queue{}, producer_fence{}, retire_fence{};
    std::uint64_t newest_sequence{}; // Can be ahead of the completed API snapshot selected for pixels.
    // Frozen under the same lock/clock as selection; diagnostics only. A zero
    // tick is unavailable, never evidence that a source has age zero.
    std::uint64_t selection_tick_ms{}, newest_view_tick_ms{}, source_tick_ms{};
    std::uint32_t active_view_count{};
    std::uint64_t requested_queue{}, consumer_queue{};
    std::uint64_t producer_completed{}; // producer_fence is the required value.
    bool producer_completion_valid{}, producer_recording_retired{};
    bool finished{}, success{}, submitted{}, invalid{};
    bool repeated_frame{}; // Latest successful frame was already copied for effects.
    bool pending_frame{}; // Same FG source awaits nomination/result/submission/immutability; capture_id may still be zero.
    std::uint64_t source_id{};
    std::uint32_t viewport{};
    sunshine_scene_depth::provider_kind provider{sunshine_scene_depth::provider_kind::streamline};
  };
  enum class consumer_status {
    not_attempted, ready, missing_input, unsupported_interface, observer_not_ready,
    missing_cookie, unsupported_command_type, missing_device, missing_device_identity,
    device_mismatch, recording_missing, recording_closed, recording_invalid,
    recording_render_pass, slot_missing, ownership_mismatch, consumer_capacity,
    capture_not_ready, invalid_destination
  };
  enum class recording_loss {
    none, global_observation_loss, wildcard_alias, source_identity_unavailable,
    source_state_capacity, close_failed, opaque_commands
  };
  enum class record_stage {
    not_attempted, inactive, malformed_input, missing_source, unsupported_lifetime, unsupported_proof,
    command_interface, source_identity, observer_coverage, command_type, command_device, device_identity,
    source_lifetime, recording_missing, recording_closed, recording_invalid, recording_render_pass,
    incomplete_state, missing_state, conflicting_state, unsupported_state, resource_region,
    capacity, texture_allocation, descriptor_allocation, recorded,
    source_interface, source_device, source_description, source_cookie, retained
  };
  // This exact record call's result. Global last_status may already describe a
  // different evaluation or acquisition when the caller resumes.
  struct record_diagnostic {
    status result{status::unavailable};
    record_stage stage{record_stage::not_attempted};
    recording_loss loss{recording_loss::none};
    std::uint64_t command{}, recording_cookie{}, resource{}, device_identity{}, expected_device_identity{};
    std::uint64_t expected_generation{}, current_generation{};
    std::uint32_t width{}, height{}, format{}, flags{}, native_state{}, observed_state{};
    // Selected transition/restore basis, distinct from the unchanged raw hint.
    // Known means admission selected a supported state, not that a copy ran.
    std::uint32_t copy_state{};
    std::uint32_t dimension{}, mip_levels{}, array_size{}, samples{};
    std::uint32_t command_type{0xffffffffu};
    bool recording_closed{}, recording_invalid{}, render_pass{}, observed{}, blocked{};
    bool copy_state_known{}, used_observed_state{};
  };
  struct consumer_diagnostic {
    consumer_status result{consumer_status::not_attempted};
    recording_loss invalidation{recording_loss::none};
    std::uint64_t cookie{}, device_identity{}, expected_device_identity{};
    std::uint32_t tracked_commands{};
    bool slot_invalid{};
  };

  // Lifecycle/hook discovery must be called with actual live native D3D12
  // interfaces. Hook installation is deferred to poll(), outside callbacks.
  void initialize(bool enabled);
  void shutdown();
  bool active();
  void observe_command(std::uint64_t command);
  void command_destroyed(std::uint64_t command);
  void observe_queue(std::uint64_t queue);
  void retire_queue(std::uint64_t queue);
  void poll();
  using preservation_available_callback = bool (*)(std::uint64_t native, std::uint64_t resource_id, std::uint64_t device_id);
  // Capability lookup only; no GPU commands. Invoked without the capture mutex
  // so the ReShade resource registry can take its own lock safely.
  void set_preservation_available(preservation_available_callback callback);
  // Observing an API attempt discovers its command list, never claims ownership.
  // Only acquire of a successful, submitted current source establishes a
  // provider on that exact queue. The first accepted provider remains selected
  // through missing frames until release/reset, except that an explicit FG
  // selection policy requires its SL input. If two ordinary providers are
  // initially ready, the earlier capture wins.
  void observe_provider(std::uint64_t command);
  bool provider_active(std::uint64_t queue);
  bool provider_identity(std::uint64_t queue, sunshine_scene_depth::provider_kind &provider,
    std::uint64_t &source_id);
  // An explicit successful feature release is different from a missing frame.
  // It removes only this logical source, retaining in-flight GPU allocations
  // until their existing fence/recording retirement conditions are satisfied.
  void retire_source(sunshine_scene_depth::provider_kind provider, std::uint64_t epoch, std::uint64_t source_id);

  source_ref retain_source(std::uint64_t resource, record_diagnostic *diagnostic = nullptr);
  std::uint64_t source_present_generation(const source_ref &source);
  // BEGIN actual native Present: expires future uses of this device's old tags,
  // never an already-copied packet. No SL/native frame-number equivalence.
  void present(std::uint64_t queue);
  // Every observed DLSS attempt advances this watermark, including malformed or
  // failed attempts, so an older success cannot conceal a newer invalid frame.
  // An explicit adapter role change may supersede one source for this viewport;
  // other viewports and all existing GPU retirement obligations remain intact.
  void begin_evaluation(std::uint64_t epoch, std::uint64_t sequence, std::uint32_t viewport,
    sunshine_scene_depth::provider_kind provider = sunshine_scene_depth::provider_kind::streamline,
    std::uint64_t source_id = 0, std::uint64_t superseded_source_id = UINT64_MAX);
  // Observe before the original DLSS evaluation on its actual command list.
  // Registered preserved sources create a metadata-only ticket; other sources
  // record an independent copy. Every nonzero ticket must always finish.
  std::uint64_t record(std::uint64_t command, const input &value, record_diagnostic *diagnostic = nullptr);
  // Source adapters supply identity/metadata and always snapshot at the API
  // call, even for a registered ReShade DSV. Matching its resource identity
  // cannot authenticate a generic before-clear copy's contents. Missing state
  // hints are resolved against observed command state, never invented.
  std::uint64_t nominate(std::uint64_t command, const input &value, record_diagnostic *diagnostic = nullptr);
  // Publish validated adapter candidates and their in-progress nomination as one
  // observation. Until the candidate batch resolves, only a bounded whole-FG-pair hold may
  // use this logical identity; no previous depth becomes current pixels. Native
  // validation and GPU ordering remain shared with nominate(). Rejection clears
  // only this attempt; missing adapter input must use begin_evaluation() instead.
  std::uint64_t nominate_evaluation(std::uint64_t command, const input &value,
    std::uint64_t superseded_source_id = UINT64_MAX, record_diagnostic *diagnostic = nullptr);
  // One or two priority-ordered tags from the exact same logical evaluation.
  // Keep source authority when all captures fail, but try usable alternatives.
  std::uint64_t nominate_evaluation(std::uint64_t command, const input *candidates, std::size_t count,
    std::uint64_t superseded_source_id = UINT64_MAX, record_diagnostic *diagnostic = nullptr);
  void finish(std::uint64_t ticket, bool successful, capture_failure failure = capture_failure::evaluation_failed);

  struct preservation_ticket {
    std::uint64_t id{}, texture{}, resource_id{};
    // CPU frame/command statistics retain this lease until they discard the
    // ticket. GPU retirement is still independently owned by the capture pool.
    std::shared_ptr<const texture_reference> ownership;
    explicit operator bool() const { return id && ownership; }
  };
  struct diagnostic_ticket {
    std::uint64_t id{};
    std::shared_ptr<const texture_reference> ownership;
    explicit operator bool() const { return id && ownership; }
  };
  struct diagnostic_texture {
    std::shared_ptr<const texture_reference> ownership;
    // The handle belongs to ownership. Duplicate it for IPC; do not close it.
    // Pixels are immutable, complete and in COMMON when acquisition succeeds.
    std::uint64_t texture{}, shared_handle{}, device{}, device_identity{}, resource_id{};
    std::uint64_t capture_id{}, producer_queue{}, producer_fence{}, producer_completed{};
    std::uint32_t width{}, height{}, format{};
    sunshine_scene_depth::extent area;
    status result{status::unavailable};
    capture_failure failure{capture_failure::none};
    bool producer_recording_retired{};
  };
  // Auxiliary snapshots use the same state/recording/fence owner as depth, but
  // an independent bounded pool. They never nominate or change a depth source.
  // Call at the authenticated API boundary, before its original call. Every
  // accepted ticket must finish; retain it until acquire or explicit cancel.
  diagnostic_ticket record_diagnostic_texture(std::uint64_t command, const input &value,
    record_diagnostic *diagnostic = nullptr);
  enum class local_texture_state_policy { source_contract, prefer_observed_recording };
  // Local consumers use the same copy/fence owner, but their retired storage
  // may be reused. These tickets expose no IPC handle. Externally shared dump
  // textures above remain immutable even after the host acknowledges opening.
  // prefer_observed_recording is restricted to synchronous Streamline FG input
  // color snapshots. It prefers the known, unblocked, nonzero state on this
  // exact recording, despite a stale hint. Only absent observation permits a
  // supported, explicit nonzero declaration compatible with resource flags.
  // Incomplete/blocked/COMMON evidence never falls back to a declaration.
  diagnostic_ticket record_local_texture(std::uint64_t command, const input &value,
    record_diagnostic *diagnostic = nullptr,
    local_texture_state_policy state_policy = local_texture_state_policy::source_contract);
  void finish_diagnostic_texture(const diagnostic_ticket &ticket, bool successful);
  // Nonblocking: pending work, even on the same queue, is not shareable. Both
  // GPU completion and Reset/destruction of its producing recording are needed
  // before a foreign process can safely read the immutable snapshot.
  status acquire_diagnostic_texture(const diagnostic_ticket &ticket, diagnostic_texture &out);
  // Copy a completed immutable auxiliary snapshot into a same-format/shape
  // texture on one consumer queue. Both resource states are restored. Retain
  // source AND destination until recorded consumers and their GPU fences retire;
  // Reset alone never permits reuse. Never adds a queue wait. A ticket cannot
  // switch destination or consumer queue after its first successful copy.
  bool copy_diagnostic_texture(std::uint64_t command, std::uint64_t consumer_queue,
    const diagnostic_ticket &ticket, std::uint64_t destination, std::uint32_t destination_state,
    consumer_diagnostic *diagnostic = nullptr);
  // Revokes the ticket, never its outstanding GPU obligations. Drop all leases
  // after cancellation/IPC acknowledgement so completed storage can be reclaimed.
  void release_diagnostic_texture(const diagnostic_ticket &ticket);
  // A legal ReShade before-clear/end-of-frame opportunity records into the same
  // snapshot pool as API opportunities. It never selects a provider or advances
  // middleware evaluation watermarks. The caller supplies the actual source
  // state at this boundary; the owner still checks command/device/shape safety.
  preservation_ticket record_preserved(std::uint64_t command, std::uint64_t source,
    std::uint32_t native_state, record_diagnostic *diagnostic = nullptr);
  // Caller must first prove this ticket is the current, unambiguous preserved
  // copy for its presentation. That existing game/ReShade ordering contract is
  // distinct from native independent-queue completion. The owner checks actual
  // submission (or an earlier copy in this same open recording), exact device,
  // recording, ticket and consumer lifetime. Destination starts and ends in
  // destination_state. No source handle escapes into the caller's GPU commands.
  bool copy_preserved(std::uint64_t command, std::uint64_t consumer_queue,
    const preservation_ticket &ticket, std::uint64_t destination, std::uint32_t destination_state,
    consumer_diagnostic *diagnostic = nullptr);
  // An already admitted API capture uses the identical read-lease/copy path.
  bool copy_current(std::uint64_t command, const packet &value, std::uint64_t destination,
    std::uint32_t destination_state, consumer_diagnostic *diagnostic = nullptr);

  // Independent copies require same-queue ordering or a completed private producer
  // fence on the same device with its producing recording retired. Pending foreign
  // copies remain unavailable: never insert a reverse dependency into the game's
  // queue graph or wait on the CPU. While a same-source successor is pending,
  // the newest unconsumed completed snapshot may supply its own pixels and
  // metadata within the existing freshness limit; resets/interruptions revoke it.
  // A copy binds to one consumer queue to serialize consumer resource-state
  // transitions. Effects require an unconsumed submitted source frame; a
  // failed copy may retry, while complete_frame commits successful consumption.
  // An API frame number is never treated as a present number.
  // Diagnostics describe this exact acquisition and its first terminal failure,
  // including a rejected latest slot when no output packet can be returned.
  // A true result selects valid current source metadata, independent of pixel
  // readiness. Inspect pixel_ready for native snapshots; shared preservation
  // resolves its current copy separately. Diagnostics retain the pixel status.
  bool acquire(std::uint64_t queue, std::uint64_t present, packet &out, capture_diagnostic *diagnostic = nullptr,
    selection_policy policy = {});
  // Rendering consumes this explicit decision. The overload above is a
  // compatibility adapter and computes the same decision without returning it.
  bool acquire(std::uint64_t queue, std::uint64_t present, packet &out, acquisition_decision &decision,
    capture_diagnostic *diagnostic = nullptr, selection_policy policy = {});
  // Commit only after the selected snapshot was successfully copied for this
  // effects pass. Nomination/acquisition alone must not consume a frame.
  void complete_frame(const packet &value, std::uint64_t present);
  // Register BEFORE recording any reads into this exact command list. Failure
  // means the caller must not issue a read. The native post-Execute hook retires
  // the registered recording with a queue fence.
  // It is invalid to reuse the packet in another command list without marking
  // that consumer too (or holding ownership through its independent completion
  // fence, as the sampler does). Pending foreign-queue packets remain unreadable
  // until producer completion and recording retirement; this adds no GPU wait.
  // Optional diagnostics describe this exact call, not the shared status last
  // written by another producer/consumer thread. They do not alter admission.
  bool mark_consumer(std::uint64_t command, const packet &value, consumer_diagnostic *diagnostic = nullptr);
  status last_status();
  const char *name(status value);
  const char *name(capture_failure value);
  const char *name(consumer_status value);
  const char *name(recording_loss value);
  const char *name(record_stage value);
  const char *name(selection_reason value);
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
  namespace testing { bool diagnostic_snapshot_regression(); }
  namespace testing { bool zero_cookie_submission_regression(); bool unsupported_com_boundary_regression(); bool source_cookie_reentry_regression(); bool recording_recovery_regression(); bool recording_state_loss_regression(); }
  namespace testing { bool submission_completion_regression(); bool provider_admission_regression(); bool crop_region_regression(); bool record_diagnostic_regression(); }
#endif

}
