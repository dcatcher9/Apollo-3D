// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "api_contract.h"
#include "game3d_frame_record.h"
#include "proven_state.h"
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
    // The copy is recorded inside the call that supplied the resource: a tag
    // call, or an evaluation whose own local inputs name it. Frame generation
    // presents earlier frames on another thread meanwhile; such a Present
    // cannot end this call's own lifetime (see source_lifetime_current).
    bool at_tag_call{};
    // The copy is made for the frame the tag was set for, before that frame's
    // Present: an evaluation that names the tag's own frame
    // (evaluated_depth_tag::own_frame). A tag's lifetime ends at its own
    // frame's Present (sl_core_types.h eValidUntilPresent, eValidUntilEvaluate):
    // Presents of earlier frames since the tag, which multi frame generation
    // shows meanwhile, leave it current.
    bool own_frame{};
    // A copy deferred to its frame's Present (api_contract.h deferred_copy,
    // owner ruling R2): the queue whose Present it precedes, which executes
    // the copy on ReShade's immediate list. Its declaration describes the
    // state there; the source's last proven state on this queue
    // (last_proven_state), where there is one, must agree with it.
    std::uint64_t present_queue{};
    // A nomination whose copy is refused is no nomination: no metadata-only
    // ticket is published, so the source's latest observation is unchanged.
    // DLSS-G's own evaluation (frame_generation_inputs.h) supplies FG depth
    // only through recorded copies; a refusal must change nothing.
    bool pixels_required{};
  };
  struct packet {
    input metadata;
    std::shared_ptr<const texture_reference> ownership;
    // queue is the requested consumer queue; the producer remains explicit
    // when an immutable copy is handed to a different queue on the same device.
    std::uint64_t texture{}, shader_resource{}, device{}, queue{}, producer_queue{}, capture_id{}, producer_fence{};
    std::uint64_t resource_id{};
    // The present ordinal of the frame this copy is (presented_by): the
    // display keeps it with a retained copy (retained_depth::present_stamp).
    std::uint64_t present{};
    // A successful source nomination can own this presentation while capture
    // is unsupported/pending. This authorizes neither texture access nor a
    // fallback selector.
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
    // Lost FG mode evidence revokes FG inputs, not independent ordinary depth.
    bool exclude_unconfirmed_fg{};
    // Required FG: when its options turned it on (frame_generation_scope_live).
    std::uint64_t enabled_tick{};
    // The Presents DLSS-G adds per real frame while FG is confirmed on (the
    // game's numFramesToGenerate; 1 when the count is unknown), also for a
    // silent scope; 0 without FG. A frame that made a frame-generation call
    // presents at its first Present and these (frame_presenting).
    std::uint32_t generated_presents{};
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
    queue_retired, consumer_capacity, source_retired,
    // A copy from a proven state (record_preserved at the end of the frame)
    // whose state changed before its list was executed, or which another
    // queue executed.
    proven_state_changed
  };
  // Selection evidence only; never grants GPU access to a pending snapshot.
  // format_changed: the completed snapshot's storage family, view or resource
  // kind differs from its pending successor's (a dynamic render size of the
  // same source does not refuse it).
  enum class selection_reason {
    not_attempted, no_current_nomination, inactive_views, ambiguous_views,
    pending_nomination, current_capture, current_already_consumed, no_admissible_capture,
    no_completed_snapshot, completed_before_gap, completed_not_older,
    completed_already_consumed, presentation_already_selected, format_changed,
    completed_not_readable, completed_snapshot, fg_scope_missing, fg_scope_mismatch,
    // A viewport's FG input is pending, or a missing frame (it failed or was
    // refused), with no completed snapshot, and the newest readable
    // unconsumed snapshot of the viewport's ordinary source (its display
    // lineage) stands in (select_capture).
    lineage_snapshot,
    // The other provider's readable copy was recorded before the owner's
    // last selected copy (capture ids follow publication order): ownership
    // does not move to older depth (select_provider).
    completed_older_than_shown
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
    // The newest evaluated frame's present ordinal while this Present still
    // presents that frame (frame_presenting), else 0: a retained copy of
    // that frame or a later one is its own depth here (decide_display).
    std::uint64_t presenting_floor{};
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
    bool pending_frame{}; // Same FG source awaits result/submission/immutability, or any source's nomination is in progress; capture_id may still be zero.
    std::uint64_t source_id{};
    std::uint32_t viewport{};
    sunshine_scene_depth::provider_kind provider{sunshine_scene_depth::provider_kind::streamline};
    // The reported source is an FG scope's own input; without one, an FG
    // scope the policy requires.
    bool frame_generation_input{};
  };
  // What acquire chose for one consumer queue at its newest acquisition, and
  // why (docs/reshade-sbs.md, Frame record): the selection's own outcome,
  // frozen under the capture lock with the decision acquire returned. Nothing
  // decides from it, and it never authorizes pixels.
  struct depth_choice {
    std::uint64_t present{}, tick_ms{};
    // The reported source's role: fg_depth (an FG scope's own input, or the
    // required FG scope before one, a nomination still in progress
    // included), scene_depth, or none without a source or a required scope.
    sunshine_game3d::source_slot rung{sunshine_game3d::source_slot::none};
    acquisition_decision decision;
    status result{status::unavailable};
    selection_reason selection{selection_reason::not_attempted};
    capture_failure failure{capture_failure::none};
    bool pixel_ready{};
    // The policy it selected under.
    bool fg_required{}, fg_excluded{};
    // The queue's provider after this acquisition: ownership outlives one
    // Present (protected owner, association).
    bool owner_established{};
    sunshine_scene_depth::provider_kind owner{sunshine_scene_depth::provider_kind::streamline};
    std::uint64_t owner_source{};
  };
  enum class consumer_status {
    not_attempted, ready, missing_input, unsupported_interface, observer_not_ready,
    missing_cookie, unsupported_command_type, missing_device, missing_device_identity,
    device_mismatch, recording_missing, recording_closed, recording_invalid,
    recording_render_pass, slot_missing, ownership_mismatch, consumer_capacity,
    capture_not_ready, invalid_destination
  };
  enum class recording_loss {
    none, global_observation_loss, source_identity_unavailable,
    source_state_capacity, close_failed
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
    // Scoped to the exact recording above. Aggregate observer notifications
    // elsewhere do not prove that this list transitioned the nominated source.
    std::uint64_t native_barrier_calls{}, native_transition_count{}, tracked_source_count{};
    std::uint64_t last_barrier_command{}, source_cookie{};
    std::uint32_t width{}, height{}, format{}, flags{}, native_state{}, observed_state{};
    // Selected transition/restore basis, distinct from the unchanged raw hint.
    // Known means admission selected a supported state, not that a copy ran.
    std::uint32_t copy_state{};
    std::uint32_t dimension{}, mip_levels{}, array_size{}, samples{};
    std::uint32_t command_type{0xffffffffu};
    bool recording_closed{}, recording_invalid{}, render_pass{}, observed{}, blocked{};
    // The observed entry is a split barrier that began and has not ended.
    bool pending_split{};
    // The observed entry is an enhanced Barrier's layout and access (raw
    // D3D12 values; observed_state is then that layout's legacy equivalent,
    // 0xffffffff without one), and the copy used enhanced barriers.
    bool observed_enhanced{}, enhanced_copy{};
    std::uint32_t observed_layout{}, observed_access{};
    bool copy_state_known{}, used_observed_state{};
    // The copy used the SDK input contract because the recording had no entry.
    bool used_contract_state{};
    // The copy used the source's last proven state (last_proven_state)
    // because the recording had no entry.
    bool used_proven_state{};
    // The recording had no entry and an executed enhanced barrier latched the
    // source's layout (proven_state.h): its claim was checked against that
    // proof (used_proven_state) or the copy refused (the matrix's D6, N8).
    bool earlier_layout{};
  };
  // Where an auxiliary copy's pre-copy state came from (diagnostic
  // provenance; it changes no admission). The values are fixed: dumps record
  // them by name (name(state_basis)).
  enum class state_basis : std::uint8_t { unknown, observed_legacy, observed_enhanced, declared, contract, proven };
  struct consumer_diagnostic {
    consumer_status result{consumer_status::not_attempted};
    recording_loss invalidation{recording_loss::none};
    std::uint64_t cookie{}, device_identity{}, expected_device_identity{};
    std::uint32_t tracked_commands{};
    bool slot_invalid{};
    // How a read was ordered after its producer (frame_record's
    // depth_read_order: 1 same queue, 2 completed, 3 ordered by the
    // queue-order model) and the producer position it followed.
    std::uint8_t read_order{};
    std::uint64_t producer_queue{}, producer_ticket{};
  };

  // Lifecycle/hook discovery must be called with actual live native D3D12
  // interfaces. Hook installation is deferred to poll(), outside callbacks.
  void initialize(bool enabled);
  void shutdown();
  bool active();
  void observe_command(std::uint64_t command);
  void command_destroyed(std::uint64_t command);
  // ReShade reports every command list it wraps from its own proxy, whose
  // dispatch no D3D12 runtime change can move (D3D12Core 1.619 moves native
  // lists to per-object tables at Reset). This lifecycle is the only source of
  // a list's recording (Reset, Close, render passes) and of capture
  // coverage: a list is covered while it is open and outside a render pass.
  // Native barrier hooks supply observed resource states where they see the
  // list; otherwise the declared state applies. native is ReShade's native
  // object for the list.
  enum class list_event : unsigned { created, reset, closed, pass_begin, pass_end, destroyed, count };
  void observe_list_event(std::uint64_t native, list_event event);
  // ReShade's own immediate list (native) on its queue, which its list events
  // never report. ReShade records it only during present/effects events, open
  // outside any render pass, and resets it right after each submission, so
  // each submission the queue hook observes ends its recording and begins the
  // next; it is registered only while that hook is ready. A ReShade event for
  // the same object proves a game list reused it.
  void observe_runtime_list(std::uint64_t native, std::uint64_t queue);
  struct list_coverage_counts {
    // Capture admissions by ReShade's lifecycle view: covered (open, outside a
    // render pass) or not, with what the lifecycle saw instead.
    std::uint64_t covered{}, unknown{}, closed{}, pass{};
    // Of covered: the barrier hooks observed the list's states, or the game's
    // declared state applied (refused or moved method tables).
    std::uint64_t states_observed{}, states_declared{};
  };
  list_coverage_counts list_coverage();
  // These counts as one log line, at most every 5 s and only after they
  // changed; false when there is nothing to log.
  bool list_coverage_report(std::uint64_t now_ms, char *out, std::size_t size);
  // One log line naming each safety bound of the capture owner (its queue
  // places, source table and snapshot pools) that refused a capture since the
  // last line, each once per session; false when no bound is newly hit.
  bool capacity_report(char *out, std::size_t size);
  void observe_queue(std::uint64_t queue);
  // Once per Present, before any capture/copy of that Present: observe the
  // effects pass's command list, ReShade's immediate list (registering it as
  // in observe_runtime_list) and its queue with one interface check each and
  // one capture-lock round trip. Equivalent to observe_command(command),
  // observe_queue(queue) and observe_runtime_list(immediate, queue); later
  // copies of the same Present on the registered immediate list skip their
  // repeated interface checks and observation.
  void observe_present(std::uint64_t command, std::uint64_t queue, std::uint64_t immediate);
  void retire_queue(std::uint64_t queue);
  // ReShade destroys a device (device: its native object), the game having
  // released its last reference (destroy_device). The capture owner keeps no
  // reference to a device, only to its resources, which go now so that the
  // device is destroyed and the next one starts clean. Its queue places signal their
  // fences and the GPU work submitted to them is waited for, then its
  // snapshots and queue places leave the owner, and the resources of its
  // sources and snapshot storage are released wherever they are still held
  // (an adapter's tag, a packet, a ticket: those are then empty, and every
  // call refuses them). Its identity is cleared. Work that did not finish
  // within a safety bound is never released: it is kept for good. out gets
  // one log line; false when the device's objects were kept.
  bool release_device(std::uint64_t device, char *out = nullptr, std::size_t size = 0);
  void poll();
  // Observing an API attempt discovers its command list, never claims ownership.
  // Only acquire of a successful, submitted current source establishes a
  // provider on that exact queue. Readable SL takes priority over NGX; pending
  // SL cannot displace working NGX. An established SL source keeps priority
  // through a bounded pending interval, then usable NGX may replace it. Each
  // packet keeps its own camera/encoding. Explicit FG still requires SL FG.
  void observe_provider(std::uint64_t command);
  bool provider_active(std::uint64_t queue);
  // Handoff evidence only (provider hysteresis): some API provider began a
  // real evaluation (a nonzero epoch; Streamline SR's source 0 included, the
  // malformed-input marker's epoch 0 not) within window_ms. Grants nothing.
  bool evaluation_live(std::uint64_t now_ms, std::uint64_t window_ms);
  // An enabled Streamline FG scope (its observation epoch and viewport) owns
  // that viewport's depth only for the frames FG supplied: it is live while
  // its newest frame outcome is a copy of its own source
  // (fg_depth_key(viewport)) that was recorded and accepted by the API since
  // FG was turned on (enabled_tick), within the last second (the same second
  // after which a silent or rejected source loses its association). Turning
  // FG on, or a single copy, grants nothing more. Otherwise the viewport's
  // ordinary SR/RR depth may supply depth, and the next accepted FG copy
  // takes ownership back.
  bool frame_generation_scope_live(std::uint64_t epoch, std::uint32_t viewport, std::uint64_t enabled_tick);
  // A frame of that FG scope's viewport began (Streamline's per-frame
  // constants for it with a new key, frame_generation::frame_begins, which
  // the probe decides once per constants call). A live scope expects each frame to record an FG
  // copy, in flight or accepted: when the next frame begins with none
  // recorded since the previous one began, that frame had no FG copy, so
  // the scope turns silent (as frame_generation_missed records, ordered after
  // every copy recorded so far) and the viewport's SR/RR depth supplies it.
  // FG keeps a viewport only while each frame supplies it: a frame that
  // makes no FG call misses it too, and the viewport's SR/RR depth is copied
  // from the frame after the first without FG depth, ready when the held FG
  // copy expires.
  void frame_generation_frame(std::uint64_t epoch, std::uint32_t viewport, std::uint64_t enabled_tick);
  // FG options turned on at enabled_tick (0: forget it). For the source age
  // after it, copies on the queue that reads depth are copied out too, since
  // FG's first Present moves Present to DLSS-G's queue after them (U-C1b),
  // until Present is seen on another queue.
  void frame_generation_enabled(std::uint64_t enabled_tick);
  // FG is producing frames now: an FG scope's frame began
  // (frame_generation_frame calls it) or DLSS-G evaluated through NGX. The
  // first such call after FG options turned on, or after fg_silent_ms
  // without any, opens the same window for the source age, so a first FG
  // Present long after the options call (FG turned on in a menu) or with
  // no options call seen (DLSS-G seen only through NGX) is covered too.
  void frame_generation_activity(std::uint64_t now);
  // A frame of that FG scope ended without a copy (its FG tag refused, its
  // copy deferred to its Present cancelled by an untag, refused or lapsed,
  // or DLSS-G's own evaluation of it refused). sequence is the frame's
  // observation sequence (its tag call's, or the evaluation's): an accepted
  // copy of a newer frame keeps the scope live, one of an older frame does not.
  void frame_generation_missed(std::uint64_t epoch, std::uint32_t viewport, std::uint64_t sequence);
  bool provider_identity(std::uint64_t queue, sunshine_scene_depth::provider_kind &provider,
    std::uint64_t &source_id);
  // An explicit feature release or authority revocation differs from a missing frame.
  // It removes only this logical source, retaining in-flight GPU allocations
  // until their existing fence/recording retirement conditions are satisfied.
  // serial: the observation serial of the call that retired it (an FG
  // options call's; 0 none). FG Off (a Streamline FG source) retires only
  // the FG role of its viewport's display lineage, and that role stays
  // retired until frame_generation_restored: meanwhile an input of the role
  // is the viewport's ordinary source.
  void retire_source(sunshine_scene_depth::provider_kind provider, std::uint64_t epoch, std::uint64_t source_id,
    std::uint64_t serial = 0);
  // FG On for the viewport in epoch: its FG role is live again.
  void frame_generation_restored(std::uint64_t epoch, std::uint32_t viewport);
  // Whether an FG input of the viewport admitted at serial (its tag call's
  // or DLSS-G evaluation's observation serial) belongs to a retired FG role:
  // admitted before the retiring options call, or while FG stays off. Such
  // an input is the frame's own depth of the display lineage, the
  // viewport's ordinary source, never its FG input.
  bool frame_generation_retired(std::uint64_t epoch, std::uint32_t viewport, std::uint64_t serial);

  // A newly retained texture (not a buffer or simultaneous-access one) is
  // watched from then on for the enhanced-layout latch (proven_state_counts).
  source_ref retain_source(std::uint64_t resource, record_diagnostic *diagnostic = nullptr);
  std::uint64_t source_present_generation(const source_ref &source);
  // Whether the barrier hooks recorded a state of source on command's current
  // recording (what a copy recorded there would start from). False when they
  // do not observe command, or it names nothing of source. writing, when
  // given, says whether that state lets the recording write source
  // (copy_state::writes). Call on the thread recording command.
  bool recorded_state(std::uint64_t command, const source_ref &source, bool *writing = nullptr);
  // Whether the UI layer's clear-boundary copy (game3d_ui_layer.cpp
  // record_into), which leaves and restores a render target's RENDER_TARGET
  // state with legacy barriers, may run on command for resource (the
  // contract matrix's D12). Not for a simultaneous-access target, which may
  // still be in COMMON there, nor for one in an enhanced layout: the latest
  // barrier on command's recording, or, when that names nothing, the
  // enhanced-layout latch (proven_state.h; the target is watched from its
  // first query). A legacy state on the recording must be RENDER_TARGET. With
  // the capture owner inactive only the access flag decides. Call on the
  // thread recording command.
  bool legacy_render_target_copy(std::uint64_t command, std::uint64_t resource);
  // Executions of game lists observed so far, on any queue: each
  // ExecuteCommandLists that submitted a list with a recording other than
  // ReShade's own immediate lists (DXGI's internal Present list has none). A
  // copy deferred to its frame's Present needs one since its tag call: the
  // game submits a frame's work before presenting it, and a Present with
  // none since is an earlier frame's (multi frame generation presents them
  // while the game records).
  std::uint64_t game_executions();
  // The newest of those executions (its game_executions() count) on queue,
  // as the submission hook reports it; 0 when none was recorded there. A
  // full record table gives a queue the entry of the queue whose newest
  // execution is oldest, which records again at its next execution, and a
  // retired queue's entry is freed. Under frame generation a deferred copy
  // without DLSS-G's evaluations needs one on its presenting queue.
  std::uint64_t game_executions_on(std::uint64_t queue);
  // The ordinal of the Present that presents the frame being recorded now
  // (one more than the Presents counted so far): a deferred copy's tag call
  // and DLSS-G's evaluation are stamped with it, and present_deferred, which
  // runs before present() counts its Present, reads that Present's own.
  std::uint64_t next_present_ordinal();
  // An evaluation call that begins no view (a live FG scope's SR/RR
  // evaluation, which the probe leaves to the scope): a later frame's call,
  // which ends an earlier frame's presenting window as begin_evaluation
  // does (frame_presenting), so a Present after it cannot take that frame's
  // copy, past the source age, for its own.
  void evaluation_called(std::uint64_t epoch);
  // A retained source's allocation width and height (false without one).
  bool source_extent(const source_ref &source, std::uint32_t &width, std::uint32_t &height);
  // BEGIN actual native Present: expires future uses of this device's old tags,
  // never an already-copied packet. No SL/native frame-number equivalence.
  void present(std::uint64_t queue);
  // Every observed DLSS attempt advances this watermark, including malformed or
  // failed attempts, so an older success cannot conceal a newer invalid frame:
  // it is never that frame's current capture, only, within its own checks, a
  // missing frame's completed stand-in (select_capture).
  // An explicit adapter role change may supersede one source for this viewport;
  // other viewports and all existing GPU retirement obligations remain intact.
  void begin_evaluation(std::uint64_t epoch, std::uint64_t sequence, std::uint32_t viewport,
    sunshine_scene_depth::provider_kind provider = sunshine_scene_depth::provider_kind::streamline,
    std::uint64_t source_id = 0, std::uint64_t superseded_source_id = UINT64_MAX);
  // An evaluation whose copy its caller nominates next (nominate): begin()
  // begins it as begin_evaluation does and, under the same lock, publishes
  // its nomination as in progress (nominate_evaluation's) until end() or the
  // scope's end, whichever path its caller returns by. NGX reads its
  // parameters and retains its source between the two. An acquisition
  // meanwhile, another thread's Present, does not read the attempt as
  // refused, which moved acquire's gap watermark and so revoked the source's
  // earlier snapshots still in flight: the snapshot completed before it
  // stands in, and without one the acquisition holds. A refused attempt,
  // once the nomination ends, is a missing frame: it moves no gap either
  // (since 10-28), and the same snapshot stands in.
  class evaluation_nomination {
  public:
    evaluation_nomination() = default;
    evaluation_nomination(const evaluation_nomination &) = delete;
    evaluation_nomination &operator=(const evaluation_nomination &) = delete;
    ~evaluation_nomination() { end(); }
    void begin(std::uint64_t epoch, std::uint64_t sequence, std::uint32_t viewport,
      sunshine_scene_depth::provider_kind provider, std::uint64_t source_id,
      std::uint64_t superseded_source_id = UINT64_MAX);
    void end();
    // depth_capture::nominate for this nomination's copy, published at
    // begin, without a second capture-lock round trip to find it published.
    // Another value (or none begun) takes nominate's own path.
    std::uint64_t nominate(std::uint64_t command, const input &value, record_diagnostic *diagnostic = nullptr);
  private:
    struct {
      sunshine_scene_depth::provider_kind provider{};
      std::uint64_t epoch{}, sequence{}, source_id{};
      std::uint32_t viewport{};
      bool pending{};
    } nominated;
  };
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
  // Test only: observe before the original DLSS evaluation on its actual
  // command list and record an independent copy, without nominating it.
  // Every nonzero ticket must always finish.
  std::uint64_t record(std::uint64_t command, const input &value, record_diagnostic *diagnostic = nullptr);
#endif
  // Source adapters supply identity/metadata and always snapshot at the API
  // call, even for a registered ReShade DSV. Matching its resource identity
  // cannot authenticate a Generic end-of-frame copy's contents. Missing state
  // hints are resolved against observed command state, never invented. The
  // copy of its provider's newest evaluation (begin_evaluation) is that
  // evaluation's nomination in progress, as nominate_evaluation's is; an
  // evaluation_nomination publishes it from the evaluation's begin instead.
  std::uint64_t nominate(std::uint64_t command, const input &value, record_diagnostic *diagnostic = nullptr);
  // Publish validated adapter candidates and their in-progress nomination as one
  // observation. Until the candidate batch resolves (any evaluation's, SR's as
  // well as FG's), an acquisition takes the source's newest completed,
  // unconsumed snapshot that the pending or missing frame would take
  // (select_capture), else holds; no earlier snapshot of the source is
  // revoked (acquire's gap watermark, which a refused or failed attempt does
  // not move either). Native
  // validation and GPU ordering remain shared with nominate(). Rejection clears
  // only this attempt; missing adapter input must use begin_evaluation() instead.
  std::uint64_t nominate_evaluation(std::uint64_t command, const input &value,
    std::uint64_t superseded_source_id = UINT64_MAX, record_diagnostic *diagnostic = nullptr);
  // Up to three priority-ordered tags from the exact same logical evaluation.
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
    // Strict diagnostic acquisition exposes complete immutable COMMON pixels.
    // Local acquisition can instead authorize an ordered same-queue GPU copy.
    std::uint64_t texture{}, shared_handle{}, device{}, device_identity{}, resource_id{};
    std::uint64_t capture_id{}, producer_queue{}, producer_fence{}, producer_completed{};
    std::uint32_t width{}, height{}, format{};
    sunshine_scene_depth::extent area;
    status result{status::unavailable};
    capture_failure failure{capture_failure::none};
    bool producer_recording_retired{};
    // The pre-copy state's provenance, reported even before the pixels are
    // admitted.
    state_basis basis{state_basis::unknown};
  };
  // Auxiliary snapshots use the same state/recording/fence owner as depth, but
  // an independent bounded pool. They never nominate or change a depth source.
  // Call at the authenticated API boundary, before its original call. Every
  // accepted ticket must finish; retain it until acquire or explicit cancel.
  enum class texture_state_policy { source_contract, prefer_observed_recording };
  // Preserve the declared lifetime when choosing a policy. A longer-lived SL
  // input may describe its state at later consumption, not this tag boundary.
  inline texture_state_policy auxiliary_state_policy(const input &value) {
    return value.provider == sunshine_scene_depth::provider_kind::streamline &&
      value.valid_until == sunshine_scene_depth::lifetime::at_call ?
      texture_state_policy::prefer_observed_recording : texture_state_policy::source_contract;
  }
  // Colour view families and the typed format a UI snapshot is read as
  // (api_contract.h).
  using sunshine_api_contract::auxiliary_snapshot_format;
  using sunshine_api_contract::color_family;
  using sunshine_api_contract::same_color_family;
  using sunshine_api_contract::typeless_color;
  // prefer_observed_recording is restricted to at-call Streamline auxiliary
  // snapshots. UI adapters use it for both local and shared copies, independently
  // of FG mode or storage lifetime. It prefers the known, unblocked, nonzero state on this
  // exact recording, despite a stale hint. Only absent observation permits a
  // supported, explicit nonzero declaration compatible with resource flags.
  // Incomplete/blocked/COMMON evidence never falls back to a declaration.
  diagnostic_ticket record_diagnostic_texture(std::uint64_t command, const input &value,
    record_diagnostic *diagnostic = nullptr,
    texture_state_policy state_policy = texture_state_policy::source_contract);
  // Local consumers use the same state/copy/fence owner, but their retired storage
  // may be reused. These tickets expose no IPC handle. Externally shared dump
  // textures above remain immutable even after the host acknowledges opening.
  diagnostic_ticket record_local_texture(std::uint64_t command, const input &value,
    record_diagnostic *diagnostic = nullptr,
    texture_state_policy state_policy = texture_state_policy::source_contract);
  void finish_diagnostic_texture(const diagnostic_ticket &ticket, bool successful);
  // Nonblocking: pending work, even on the same queue, is not shareable. Both
  // GPU completion and Reset/destruction of its producing recording are needed
  // before a foreign process can safely read the immutable snapshot.
  status acquire_diagnostic_texture(const diagnostic_ticket &ticket, diagnostic_texture &out);
  // Local-only snapshots may be copied after their successful producer
  // submission on that same queue, using GPU order without a CPU wait or Reset.
  // Foreign queues retain completion/producer-retirement requirements. Shared
  // dump tickets cannot opt into this path; GPU retirement is never relaxed.
  status acquire_local_texture(const diagnostic_ticket &ticket, std::uint64_t consumer_queue,
    diagnostic_texture &out);
  // Copy a completed immutable auxiliary snapshot into a same-format/shape
  // texture on one consumer queue. Both resource states are restored. Retain
  // source AND destination until recorded consumers and their GPU fences retire;
  // Reset alone never permits reuse. Never adds a queue wait. A ticket cannot
  // switch destination or consumer queue after its first successful copy.
  // immediate: command is the consumer queue's ReShade immediate list (see
  // copy_current), which may fall back when its hooks were refused.
  bool copy_diagnostic_texture(std::uint64_t command, std::uint64_t consumer_queue,
    const diagnostic_ticket &ticket, std::uint64_t destination, std::uint32_t destination_state,
    consumer_diagnostic *diagnostic = nullptr, bool immediate = false);
  bool copy_local_texture(std::uint64_t command, std::uint64_t consumer_queue,
    const diagnostic_ticket &ticket, std::uint64_t destination, std::uint32_t destination_state,
    consumer_diagnostic *diagnostic = nullptr, bool immediate = false);
  // Direct binding of a local auxiliary snapshot (no second copy). A shader
  // view of the snapshot's own storage, valid for reads recorded into command
  // on this Present.
  struct local_view {
    // A D3D12 CPU descriptor handle in a non-shader-visible heap owned by the
    // snapshot storage; on ReShade D3D12 this is the reshade::api::
    // resource_view handle to pass to push_descriptors (descriptors are
    // copied at record time). It is not one of ReShade's own views, so
    // device::get_resource_from_view does not resolve it: use resource.
    std::uint64_t view{};
    std::uint64_t resource{}; // ID3D12Resource * (reshade::api::resource handle).
    std::uint64_t capture_id{};
    std::uint32_t width{}, height{}, format{}, view_format{};
  };
  // Leases ticket's snapshot to command, ReShade's registered immediate list
  // (observe_runtime_list/observe_present) of consumer_queue, open outside any
  // render pass. Records the snapshot's COMMON -> PIXEL|NON_PIXEL_SHADER_RESOURCE
  // transition there (once per recording) and registers command's recording as
  // a consumer exactly like copy_local_texture, so the slot is neither reused
  // nor released until that submission's queue fence passes (Reset alone
  // never retires it). view_format is the typed DXGI format the consumer reads
  // (same typeless family as the storage); the SRV is created on first use and
  // cached with the storage. A ticket offered again on a later Present takes a
  // new lease on that Present's list.
  // False (nothing recorded; the caller keeps copy_local_texture) for D3D11 or
  // a list that is not the registered immediate list, a producer on another
  // queue (foreign or mixed), a shared/strict diagnostic ticket, a ticket that
  // was already copied, an unadmitted or released ticket, no consumer capacity,
  // or an incompatible view format. Never allocates, never waits.
  bool lease_local_view(std::uint64_t command, std::uint64_t consumer_queue, const diagnostic_ticket &ticket,
    std::uint32_t view_format, local_view &out, consumer_diagnostic *diagnostic = nullptr);
  // Once per Present after the Present's last read of any leased view (before
  // the list is submitted): records the transitions of every snapshot leased
  // on command's current recording back to COMMON. Snapshots stay leased
  // (unreusable) until their GPU retirement as above.
  void end_local_views(std::uint64_t command);
  // Revokes the ticket, never its outstanding GPU obligations. Drop all leases
  // after cancellation/IPC acknowledgement so completed storage can be reclaimed.
  void release_diagnostic_texture(const diagnostic_ticket &ticket);
  // A legal ReShade end-of-frame opportunity records into the same snapshot
  // pool as API opportunities. It never selects a provider or advances
  // middleware evaluation watermarks. The caller supplies the contract
  // (api_contract.h generic): at the end of the frame the source's last proven
  // state on queue, the queue that executes command (end_of_frame,
  // last_proven_state), or a declared state (preserved); a state observed on
  // command's own recording takes precedence. The owner still checks command/device/shape
  // safety, and a copy from a proven state is discarded at submission when
  // the state changed meanwhile or command runs on another queue.
  preservation_ticket record_preserved(std::uint64_t command, std::uint64_t source,
    const sunshine_api_contract::contract &boundary, std::uint64_t queue = 0, record_diagnostic *diagnostic = nullptr);
  // A declared state (generic::preserved).
  preservation_ticket record_preserved(std::uint64_t command, std::uint64_t source,
    std::uint32_t native_state, record_diagnostic *diagnostic = nullptr);

  // The per-resource last-proven state (proven_state.h): the state a texture
  // is proven to be in after the work executed on its device. A texture is
  // tracked from its creation (ReShade's init_resource): initial_state is its
  // native creation state, UINT32_MAX when ReShade names none. Buffers and
  // simultaneous-access textures decay to COMMON after each execution and
  // are not tracked. Then each executed list whose recording the barrier
  // hooks fully observed proves the state its last transition or enhanced
  // barrier left a tracked texture in, in submission order; an unobserved
  // execution, observer loss, an aliasing barrier naming the texture (or
  // none: any placed resource) or a split or blocked barrier leaves it
  // unproven until a later barrier proves it again. A proof holds only for
  // the queue that executed it (a creation state for every queue until a list
  // uses the texture, unless enhanced barriers were seen).
  // Any enhanced barrier of an executed list that names a texture held
  // (tracked, or an adapter's source watched from its retention) latches its
  // layout for good: a copy of it whose recording names nothing takes no
  // legacy barrier from a claim. It copies only from a shader-read claim's
  // SHADER_RESOURCE layout as the texture's proof from that list states it,
  // with enhanced barriers and checked at execution as a proven copy is;
  // anything else is refused (the contract matrix's D6 and N8).
  using proven_state = sunshine_proven_state::state;
  void resource_created(std::uint64_t resource, std::uint32_t initial_state);
  // A depth-stencil texture ReShade destroyed, by the identity
  // resource_created tracked it under (depth_addon's native identity): the
  // table forgets it and its latch before it next holds a texture, so
  // destroyed textures take no place and lose no latch. Takes no lock and
  // never touches the dying object.
  void resource_destroyed(std::uint64_t identity);
  // A source whose copy a tag call deferred to its frame's Present
  // (input::present_queue) is tracked from then on, unproven until an
  // executed list proves it, so that copy starts from the state the game's
  // work left it in on that queue (an enhanced layout included) or is refused
  // when that state disagrees with the declaration. A texture tracked already
  // (a depth-stencil one, from its creation) keeps its state; buffers and
  // simultaneous-access textures are not tracked. Generic's depth-stencil
  // source (record_preserved) is tracked again the same way after a full
  // table forgot it.
  void track_deferred_source(const source_ref &source);
  // A depth clear of resource's subresource 0 recorded on command, a list
  // ReShade's lifecycle reports (ClearDepthStencilView with the depth flag, or
  // a render pass that clears depth): D3D12 requires DEPTH_WRITE there, so
  // when command executes it proves that state unless a later barrier on it
  // decides. Call on the thread recording command, before the native clear.
  void observe_depth_clear(std::uint64_t command, std::uint64_t resource);
  // The texture's proven state for a copy that queue executes; false (and an
  // empty out) when it is unproven there.
  bool last_proven_state(std::uint64_t resource, std::uint64_t queue, proven_state &out);
  struct proven_state_counts {
    // The textures held: tracked, or watched from an adapter's retention
    // (retain_source); of those, latched by an executed enhanced barrier, and
    // whether a latch was lost: a full table forgot a latched one while its
    // record of forgotten latched textures was full (proven_state.h).
    std::uint64_t tracked{}, latched{};
    bool latch_lost{};
    // Executed lists that left every texture unproven, by why the hooks did
    // not observe them: no recording state (never reported or associated),
    // not reported by ReShade's lifecycle, invalidated, Reset without that
    // lifecycle, or a method table the barrier hooks do not cover.
    std::uint64_t unrecorded{}, unreported{}, invalidated{}, missed_reset{}, uncovered{};
    std::uint64_t changed_copies{};
    // Copies of a latched source whose recording named nothing: recorded
    // with enhanced barriers from its proof, or refused (the matrix's D6, N8).
    std::uint64_t earlier_copied{}, earlier_refused{};
    // Copy-outs on the producer queue (docs/reshade-sbs.md): made, and
    // copies left to the Reset rule with every list of the queue's ring in
    // flight (ring_busy) or a list call failed. U-C1b: copies recorded with
    // copy-out storage because their list was not known to execute on the
    // reader's queue (predicted), copies given an idle slot's cached storage
    // at their submission (taken), copies submitted on another queue with no
    // storage at all, left to the Reset rule (unstored), and predicted
    // storage released unused by the idle sweep. Copies that sought no
    // copy-out storage of their own because an allocation of their shape
    // failed within the source age (skipped).
    std::uint64_t copied_out{}, copy_out_busy{}, copy_out_failed{};
    std::uint64_t copy_out_predicted{}, copy_out_taken{}, copy_out_unstored{}, copy_out_released{}, copy_out_skipped{};
    // Changes of the queue that reads depth (Present seen on another queue,
    // as at FG On and Off), and those whose first acquisition there had no
    // ready pixels: a flat Present the copy-outs did not prevent.
    std::uint64_t reader_moves{}, reader_move_unreadable{};
    std::uint64_t unobserved() const { return unrecorded + unreported + invalidated + missed_reset + uncovered; }
  };
  proven_state_counts proven_counts();
  // The queue-order status line (docs/reshade-sbs.md, Queue order), into
  // out, when due: at most every 10 s while its counts change, and once when
  // QueueFenceOrder=0. Called from one thread (the probe's poll).
  bool queue_order_line(std::uint64_t now, char *out, std::size_t size);
  // O1 (docs/reshade-sbs.md, Queue order): whether capture was read in
  // flight on a fence that a late conflict poisoned since (late_conflict);
  // its presentation copy is then not held any further.
  bool read_conflicted(std::uint64_t capture);
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
  // immediate asserts that command is the consumer queue's ReShade immediate
  // list, open outside a render pass: the runtime submits it on that queue and
  // resets it without replay. If its hooks were refused, its read is retired by
  // that observed submission and the queue fence; a covered list keeps the
  // recording lease, which also verifies it is open and outside a render pass.
  bool copy_current(std::uint64_t command, const packet &value, std::uint64_t destination,
    std::uint32_t destination_state, consumer_diagnostic *diagnostic = nullptr, bool immediate = false);

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
  // readiness. Inspect pixel_ready. Diagnostics retain the pixel status.
  bool acquire(std::uint64_t queue, std::uint64_t present, packet &out, capture_diagnostic *diagnostic = nullptr,
    selection_policy policy = {});
  // Rendering consumes this explicit decision. The overload above is a
  // convenience overload for tests and callers that do not need the
  // decision; it computes the same decision without returning it.
  bool acquire(std::uint64_t queue, std::uint64_t present, packet &out, acquisition_decision &decision,
    capture_diagnostic *diagnostic = nullptr, selection_policy policy = {});
  // Commit only after the selected snapshot was successfully copied for this
  // effects pass. Nomination/acquisition alone must not consume a frame.
  void complete_frame(const packet &value, std::uint64_t present);
  // The newest depth_choice of this consumer queue; false before its first
  // acquisition that reached the selection.
  bool choice(std::uint64_t queue, depth_choice &out);
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
  // Test only. Register BEFORE recording any reads into this exact command list. Failure
  // means the caller must not issue a read. The native post-Execute hook retires
  // the registered recording with a queue fence.
  // It is invalid to reuse the packet in another command list without marking
  // that consumer too (or holding ownership through its independent completion
  // fence, as the sampler does). Pending foreign-queue packets remain unreadable
  // until producer completion and recording retirement; this adds no GPU wait.
  // Optional diagnostics describe this exact call, not the shared status last
  // written by another producer/consumer thread. They do not alter admission.
  bool mark_consumer(std::uint64_t command, const packet &value, consumer_diagnostic *diagnostic = nullptr);
#endif
  status last_status();
  const char *name(status value);
  const char *name(capture_failure value);
  const char *name(consumer_status value);
  const char *name(recording_loss value);
  const char *name(record_stage value);
  const char *name(selection_reason value);
  const char *name(state_basis value);
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
  namespace testing { bool diagnostic_snapshot_regression(); }
  namespace testing { bool initial_recording_regression(); bool zero_cookie_submission_regression(); bool unsupported_com_boundary_regression(); bool source_cookie_reentry_regression(); bool recording_recovery_regression(); bool recording_state_loss_regression(); }
  namespace testing { bool submission_completion_regression(); bool provider_admission_regression(); bool crop_region_regression(); bool record_diagnostic_regression(); }
  namespace testing { bool live_source_admission_regression(); }
  // The state-basis provenance of an auxiliary snapshot copy.
  namespace testing { bool snapshot_basis_regression(); }
  // An executed recording's sorted aliasing marks answer aliased_after
  // exactly as the unsorted ones did.
  namespace testing { bool alias_lookup_regression(); }
  // Queue places at their safety bound: a full table of busy places refuses
  // a queue and reports the bound once; an idle place is reclaimed.
  namespace testing { bool queue_place_regression(std::uint64_t filler, std::uint64_t target); }
  // Called once (then cleared) on the recording thread while a record call
  // allocates new storage without the capture lock.
  namespace testing { void set_allocation_hook(void (*hook)(void *), void *context); }
  // The next count copy-out storage allocations of record calls fail.
  namespace testing { void fail_copy_out_allocations(unsigned count); }
  // Waits until every pooled storage release finished (release_storage).
  namespace testing { void settle_storage_releases(); }
  // Makes idle colour/mask storage older by ms (poll() releases it after 2 s).
  namespace testing { void age_idle_storage(std::uint64_t ms); }
  // Whether the snapshot of capture holds copy-out storage (slot::published).
  namespace testing { bool holds_copy_out_storage(std::uint64_t capture); }
  // The native resource of capture's copy-out storage (0 none).
  namespace testing { std::uint64_t copy_out_resource(std::uint64_t capture); }
  // release_device's bound on its wait for the device's queues (0: the
  // shipping bound).
  namespace testing { void set_release_wait(std::uint64_t ms); }
#endif

}
