// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <Windows.h>
#include <cstdint>
#include <array>
#include "streamline_camera_data.h"
#include "scene_feedback.h"
#include "streamline_command_association.h"
#include "streamline_depth_content.h"
#include "native_discard_observer.h"
#include "streamline_depth_capture.h"
#include "observation_loss_journal.h"

namespace sunshine_streamline {
  struct effects_input_observation {
    std::uint64_t runtime{}, device{}, resource{}, command{};
    std::uint32_t width{}, height{}, format{};
    bool ready{};
  };
  struct selected_depth {
    std::uint64_t resource{}, source_id{}, layout_epoch{}, frame_index{};
    std::uint32_t width{}, height{}, x{}, y{}, active_width{}, active_height{};
    bool ready{};
    commands::recording_marker capture_marker;
    std::uint64_t command_queue{};
    content::copy_snapshot depth_copy;
    std::uint64_t backup_resource{}, backup_id{};
    effects_input_observation effects_input;
  };

  // The optional diagnostic branch never changes selected depth or geometry;
  // it is enabled by ReShade.ini [SUNSHINE_DEPTH] StreamlineCameraProbe=1.
  // Independent source adapters also share these hooks and can nominate/copy
  // current depth. All branches preserve the application's original API data.
  // Discovery and deferred logging occur only from poll, outside loader callbacks.
  void initialize(HMODULE addon);
  bool enabled();
  // Lightweight supported-DLL source nomination is enabled by default. This is
  // independent of the opt-in command/content diagnostic above.
  bool source_enabled();
  // Existing source-observation invalidation counter only. An unchanged value
  // does not prove frame correspondence; a changed value revokes retained depth.
  std::uint64_t depth_observation_revision();
  // Exact-revision diagnostics only; absence never changes observation validity.
  bool query_depth_observation_loss(std::uint64_t revision, loss_diagnostics::event &out);
  void poll(const selected_depth &selected);
  // Installed hooks and their pinned modules remain alive as pure pass-through until
  // process exit. This never waits for game threads or removes a live trampoline.
  void shutdown();

  // D3D12 native identities only. Root invokes these at ReShade lifecycle,
  // reset/close/execute events and captures a marker at the ACTUAL depth copy.
  // Pre-call callbacks establish observed recording/submission order, not native
  // call success, GPU completion, depth-content identity or final-color alignment.
  void command_initialized(std::uint64_t native, std::uint64_t device, std::uint64_t cookie = 0);
  void command_destroyed(std::uint64_t native);
  void command_reset(std::uint64_t native, std::uint64_t device = 0, std::uint64_t cookie = 0);
  void command_closed(std::uint64_t native);
  void queue_initialized(std::uint64_t native, std::uint64_t device, std::uint64_t cookie = 0);
  void queue_destroyed(std::uint64_t native);
  void command_executed(std::uint64_t queue, std::uint64_t command);
  void command_secondary_executed(std::uint64_t primary, std::uint64_t secondary);
  commands::recording_marker capture_command_marker(std::uint64_t native);
  // Valid native D3D12 COM object only: queue standard DiscardResource target
  // discovery from init/reset; actual hook installation is deferred to poll.
  void observe_native_discard_command(std::uint64_t native);

  // Relative content identity within one D3D12 recording, based only on observed
  // operations. Resource lifetime IDs must be unique caller-owned identities.
  // This does not claim coverage of unobservable native commands or final color.
  void depth_resource_initialized(std::uint64_t native, std::uint64_t lifetime, std::uint64_t device, bool supported);
  void depth_resource_destroyed(std::uint64_t native, std::uint64_t lifetime);
  void depth_content_write(std::uint64_t command, std::uint64_t native, std::uint64_t lifetime = 0);
  void depth_content_invalidate(std::uint64_t command, std::uint64_t native = 0, std::uint64_t lifetime = 0);
  content::copy_snapshot record_depth_copy(std::uint64_t command, content::resource_key source, content::resource_key backup);
  content::copy_snapshot forward_depth_copy(std::uint64_t command, const content::copy_snapshot &preserved,
    content::resource_key destination);

  enum class frame_identity_kind { unavailable, v1_numeric, v2_observed_token, v2_constants_call };
  struct frame_identity {
    frame_identity_kind kind{};
    std::uint64_t generation{}, numeric{};
    std::uintptr_t token{};
    bool has_numeric{};
  };
  enum class tag_scope { active_global, explicit_frame, evaluation_local };
  struct evaluated_depth_tag {
    depth_tag value;
    tag_scope scope{};
    bool present{}, supported{};
    std::uint64_t observed_tick{}, observation_sequence{};
    content::use_snapshot content_use;
    std::uint64_t resource_lifetime{}, resource_device{};
    std::uint32_t native_state{0xffffffffu};
    bool state_requires_observation{};
    depth_capture::source_ref direct_source;
    std::uint64_t source_present_generation{};
    // PrecisionInfo maps stored texels to the depth consumed by Streamline.
    double precision_scale{1.0}, precision_bias{};
  };
  struct evaluated_color_tag {
    depth_tag value; // Shared resource/extent/lifecycle metadata only; not a depth interpretation.
    tag_scope scope{};
    bool present{}, supported{}, identity_available{};
    std::uint64_t observed_tick{}, observation_sequence{};
  };
  enum class evidence_status {
    inactive, busy, missing_evaluation, evaluation_failed, observation_lost,
    untracked_frame, missing_constants, rejected_constants, invalid_camera,
    camera_reset, missing_depth, unsupported_depth, depth_not_ready,
    source_mismatch, extent_mismatch, stale, ambiguous_viewport,
    source_associated_evaluation, command_associated_evaluation, tracked_content_evaluation
  };
  struct evaluation_snapshot {
    evidence_status status{evidence_status::missing_evaluation};
    camera_data camera;
    camera_validation projection;
    decode_status decoded{decode_status::short_buffer};
    frame_identity frame;
    std::array<evaluated_depth_tag, 3> tags{}; // raw, high-resolution raw, linear
    std::array<evaluated_color_tag, 4> colors{}; // HUDless2, scaling input3, scaling output4, v2 backbuffer53
    std::uint64_t epoch{}, sequence{}, tick{}, camera_tick{}, camera_sequence{}, loss_revision{};
    sunshine_scene_feedback::sample feedback;
    std::uintptr_t command_buffer{};
    std::uint32_t viewport{}, feature{}, matched_tag{};
    selected_depth selection;
    commands::recording_marker evaluation_recording;
    commands::association command_association;
    content::association content_association;
    // Constants and applicable tags were copied before the original evaluation.
    // Additional command/content evidence is explicit in the association fields;
    // even a match does not establish complete native mutation coverage, GPU
    // completion, ReShade presentation or final-color/jitter registration.
    bool successful_evaluation{}, frame_correlated{}, recording_stable{};
    bool tag_boundary{}; // Copied synchronously while an OnlyValidNow tag is valid.
  };
  // Returns a value copy, never live pointers or mutable internal storage. Even
  // tracked_content_evaluation is NOT permission to drive geometry: complete
  // native coverage and final-color registration remain unverified. A newer invalid/drop event,
  // failed evaluation, token reuse, unsupported inputs, or age limit fails closed.
  evidence_status query_evaluation(const selected_depth &selected, evaluation_snapshot &output,
    std::uint32_t max_age_ms = 250);
  const char *name(evidence_status value);

  struct depth_source_tag {
    depth_tag value;
    std::uint64_t resource_lifetime{}, device{};
    bool present{};
  };
  struct depth_source_snapshot {
    evidence_status status{evidence_status::missing_evaluation};
    std::uint64_t epoch{}, sequence{}, tick{}, loss_revision{};
    frame_identity frame;
    std::uint32_t viewport{};
    std::array<depth_source_tag, 2> tags{}; // Preferred display-resolution48, then raw0.
  };
  // Nomination only: no selected-depth prerequisite and no geometry/content
  // proof. The owner must consume a fresh sequence once per native present and
  // match the frozen lifetime/device/extent to a valid current captured source.
  // Streamline frame identity is never a ReShade presentation frame identity.
  evidence_status query_depth_source(depth_source_snapshot &output, std::uint32_t max_age_ms = 250);

  enum class presentation_status {
    inactive, missing_marker_function, missing_bracket, pending_marker, open_bracket,
    ended_bracket, failed_marker, untracked_frame, out_of_order, repeated_frame,
    stale, observation_lost, busy
  };
  struct presentation_snapshot {
    presentation_status status{presentation_status::missing_bracket};
    frame_identity frame;
    std::uint64_t epoch{}, loss_revision{}, marker_loss_revision{}, thread_generation{}, start_sequence{}, end_sequence{}, start_tick{}, end_tick{};
    std::uint32_t thread_id{}, last_marker{};
    bool explicit_bracket{};
  };
  // Current calling thread only. This identifies an explicit successful marker
  // bracket, never swapchain identity, final-color content or GPU completion.
  presentation_status query_current_presentation(presentation_snapshot &output, std::uint32_t max_age_ms = 250);
  const char *name(presentation_status value);

  struct frame_generation_snapshot {
    std::uint64_t epoch{}, sequence{}, tick{}, loss_revision{};
    std::uint32_t viewport{}, mode{}, generated_frames{};
    bool known{}, enabled{}, automatic{};
  };
  // Successful, version-checked game options only. Enabled means requested by
  // the game; it does not identify any particular real/generated presentation.
  enum class frame_generation_query_status { unavailable, busy, ambiguous, observed };
  bool query_frame_generation(std::uint32_t viewport, frame_generation_snapshot &output,
    frame_generation_query_status *status = nullptr);

#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
  namespace testing {
    enum class abi { unsupported, v1_1_1, v2_7_30 };
    struct targets { void *constants{}, *tag{}, *tag_for_frame{}, *evaluate{}, *new_frame_token{}, *get_feature_function{}; };
    struct counters { std::uint64_t constants{}, tags{}, evaluations{}, dropped{}, invalid{}; };
    // Real inline hooks on fixture functions, without pinning the process executable.
    bool install(abi version, const targets &functions);
    counters counts();
    bool latest(std::uint32_t viewport, std::uint64_t &resource, bool &has_camera,
      bool &explicit_same_frame, bool &same_token_address);
    bool latest_snapshot(std::uint32_t viewport, evaluation_snapshot &out);
    // Value translation only, never a post-return permission to read the source.
    bool normalized_source(std::uint32_t viewport, unsigned index, sunshine_scene_depth::frame &out);
    // Fixture only: caller guarantees no callback can be running.
    void clear();
    // Exercise fail-closed observation loss without relying on scheduler luck.
    void lose_observation();
    bool install_presentation_hooks();
    bool discover_frame_generation_hooks();
    // Deterministic CPU-only contention fixture; lock/unlock on the same thread.
    void lock_source_resources();
    void unlock_source_resources();
    unsigned waiting_source_resources();
    void lock_records();
    void unlock_records();
    void lock_records_shared();
    void unlock_records_shared();
    void lock_tokens();
    void unlock_tokens();
    bool pin_tokens(unsigned index);
    void unpin_tokens();
    // One explicitly armed fixture observation of the real pre-SDK policy.
    void arm_entry_policy();
    bool entry_policy(bool &selected, bool &admissible);
  }
#endif
}
