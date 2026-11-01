// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <Windows.h>
#include <cstdint>
#include <array>
#include "streamline_camera_data.h"
#include "scene_feedback.h"
#include "streamline_depth_capture.h"
#include "presentation_scope.h"
#include "freshness.h"
#include "observation_loss_journal.h"

namespace sunshine_streamline {
  // Source adapters share these hooks and can nominate/copy current depth.
  // All branches preserve the application's original API data.
  // Discovery and deferred logging run only through the hook host, outside loader callbacks.
  void initialize(HMODULE addon);
  // Lightweight supported-DLL source nomination is enabled by default.
  bool source_enabled();
  // Existing source-observation invalidation counter only. An unchanged value
  // does not prove frame correspondence; a changed value revokes retained depth.
  std::uint64_t depth_observation_revision();
  // Exact-revision diagnostics only; absence never changes observation validity.
  bool query_depth_observation_loss(std::uint64_t revision, loss_diagnostics::event &out);
  // The adapter owns the poll lock and all installation state. The hook host
  // holds this guard across the whole per-Present maintenance sequence.
  class poll_guard {
  public:
    ~poll_guard();
    poll_guard(const poll_guard &) = delete;
    poll_guard &operator=(const poll_guard &) = delete;
    explicit operator bool() const { return acquired_; }
  private:
    explicit poll_guard(bool acquired) : acquired_(acquired) {}
    bool acquired_;
    friend poll_guard try_poll();
  };
  poll_guard try_poll();
  bool middleware_requested_for_poll();
  void discovery_step(bool api_requested, std::uint64_t now);
  void coverage_step();
  void camera_availability_step(std::uint64_t now);
  void depth_capture_step();
  void queue_order_step(std::uint64_t now);
  void pending_installs_step();
  // Installed hooks and their pinned modules remain alive as pure pass-through until
  // process exit. This never waits for game threads or removes a live trampoline.
  void shutdown();

  // ReShade's present event on queue, before that Present (and before
  // depth_capture::present ends the tags' lifetimes): makes the copies a tag
  // call deferred to its frame's Present (owner ruling R2,
  // api_contract.h deferred_copy) on immediate, ReShade's immediate list of
  // queue, once a game list executed after the tag (an earlier Present is an
  // earlier frame's; under frame generation, while DLSS-G evaluates every
  // frame, once DLSS-G evaluated a frame after the tag:
  // frame_generation::deferral_evaluations); drops those whose frame ended.
  // True when it recorded a copy: the caller submits immediate before the
  // Present.
  bool present_deferred(std::uint64_t swapchain, std::uint64_t queue, std::uint64_t immediate);

  enum class frame_identity_kind { unavailable, v1_numeric, v2_observed_token, v2_constants_call };
  struct frame_identity {
    frame_identity_kind kind{};
    std::uint64_t generation{}, numeric{};
    std::uintptr_t token{};
    bool has_numeric{};
  };
  // A frame's key, the one identity both a viewport's frame begin
  // (store_camera, depth_capture::frame_generation_frame, through
  // frame_generation::frame_begins) and DLSS-G's depth precedence
  // (frame_generation::frame_mark) compare: a numeric frame's index plus
  // one, a minted token's generation (its mint's call sequence, unique
  // within the observation), and for constants on a token the probe did not
  // see minted that token's address, which every constants call of the
  // frame repeats; 0 when the call named no frame.
  constexpr std::uint64_t frame_key(const frame_identity &frame) {
    switch (frame.kind) {
      case frame_identity_kind::v1_numeric: return frame.numeric + 1;
      case frame_identity_kind::v2_observed_token: return frame.generation;
      case frame_identity_kind::v2_constants_call: return frame.token;
      default: return 0;
    }
  }
  enum class tag_scope { active_global, explicit_frame, evaluation_local };
  // The tag call an evaluate's untag ended (a depth tag naming no resource,
  // Streamline's untag): the viewport's newest tag of that kind that named
  // one, set before the untag. present: there is such a call.
  struct ended_depth_tag {
    depth_tag value;
    tag_scope scope{};
    bool present{}, supported{}, state_requires_observation{}, commanded{};
    // It was set for the evaluated frame (evaluated_depth_tag::own_frame).
    bool own_frame{};
    std::uint32_t native_state{0xffffffffu};
    depth_capture::source_ref direct_source;
    std::uint64_t source_present_generation{};
  };
  struct evaluated_depth_tag {
    depth_tag value;
    tag_scope scope{};
    bool present{}, supported{};
    std::uint64_t observed_tick{}, observation_sequence{};
    std::uint32_t native_state{0xffffffffu};
    bool state_requires_observation{};
    // Its tag call supplied a command list (an evaluate input's evaluation
    // always does).
    bool commanded{};
    depth_capture::source_ref direct_source;
    std::uint64_t source_present_generation{};
    // An evaluation of the frame this stored tag was set for (the matrix's
    // S7): a frame tag kept in this frame's record, or a global tag set after
    // this frame's constants, both on a minted frame token. The game has not
    // presented that frame yet, so Presents since the tag were earlier
    // frames' (generated ones under frame generation) and leave its lifetime
    // (depth_capture::input::own_frame).
    bool own_frame{};
    // PrecisionInfo maps stored texels to the depth consumed by Streamline.
    double precision_scale{1.0}, precision_bias{};
    // An untag at an evaluate: the tag call it ended.
    ended_depth_tag ended;
  };
  // 11-14, 18 and 19 were unsupported_depth, depth_not_ready, source_mismatch,
  // extent_mismatch, command_associated_evaluation and
  // tracked_content_evaluation, which nothing has set since structure refactor
  // rounds D2 and D3; 3 and 16 were evaluation_failed (none since D6: a failed
  // evaluate invalidates its copy, capture_failure::evaluation_failed) and
  // ambiguous_viewport (none since D3: two active viewports are the capture
  // owner's ambiguous_views). The other values keep their numbers.
  enum class evidence_status {
    inactive, busy, missing_evaluation, observation_lost = 4,
    untracked_frame, missing_constants, rejected_constants, invalid_camera,
    camera_reset, missing_depth, stale = 15,
    source_associated_evaluation = 17
  };
  struct evaluation_snapshot {
    evidence_status status{evidence_status::missing_evaluation};
    camera_data camera;
    camera_validation projection;
    decode_status decoded{decode_status::short_buffer};
    frame_identity frame;
    std::array<evaluated_depth_tag, 3> tags{}; // raw, high-resolution raw, linear
    std::uint64_t epoch{}, sequence{}, camera_tick{}, camera_sequence{}, loss_revision{};
    sunshine_scene_feedback::sample feedback;
    std::uintptr_t command_buffer{};
    std::uint32_t viewport{}, feature{};
    // Constants and applicable tags were copied before the original evaluation.
    // That does not establish complete native mutation coverage, GPU
    // completion, ReShade presentation or final-color/jitter registration.
    bool frame_correlated{};
    bool tag_boundary{}; // Copied synchronously while an OnlyValidNow tag is valid.
    bool frame_generation_input{}; // An independently observed enabled FG role, not implied by tagging.
    // Depth direction is a fixed convention of a viewport. Without a
    // frame-correlated camera, a recent valid camera of the same viewport still
    // supplies direction. Its depth coefficients (near/far; not FOV, jitter or
    // pose) are camera constants: once the viewport's camera repeated them in
    // steady_projection_frames consecutive valid frames they supply this
    // depth's projection too. Jitter and any per-frame claim still require the
    // frame's own camera.
    bool viewport_direction{}, viewport_projection_ready{};
    std::uint32_t viewport_inverted{};
    camera_validation viewport_projection;
  };
  const char *name(evidence_status value);

  bool query_frame_generation(std::uint32_t viewport, frame_generation_snapshot &output,
    frame_generation_query_status *status = nullptr);
  // One FG snapshot per Present (docs/reshade-sbs.md, Presentation scope):
  // take_present_fg reads the FG mode once, first thing in a Present (depth's
  // present event, for every API), updates the process's one fg_view and
  // keeps the result for that Present's thread; end_present_fg drops it at
  // the Present's end. present_frame_generation returns the snapshot taken
  // for the swapchain's Present on this thread; a consumer that finds none
  // takes its own (fail-safe, counted by present_fg_fallbacks), so it never
  // reads another Present's.
  void take_present_fg(std::uint64_t swapchain);
  void end_present_fg();
  const presentation_scope::present_fg &present_frame_generation(std::uint64_t swapchain);
  std::uint64_t present_fg_fallbacks();
  // The provider's depth acquisition of the swapchain's Present on this
  // thread names the frame it presents (present_fg::depth_frame); ignored
  // for a snapshot of another swapchain.
  void present_depth_frame(std::uint64_t swapchain, std::uint64_t frame);

  // DLSS-G's own NGX evaluation of one real frame (presentation_scope.h).
  // Its inputs belong to the one viewport whose FG options were observed
  // enabled in the current epoch, under one fresh observation sequence ordered
  // with every Streamline call. DLSSG.Depth becomes that viewport's FG source
  // (fg_depth_key(viewport)) only through a copy recorded on the evaluation's own
  // command list in a state observed there; its accepted copies keep the FG
  // scope live exactly as accepted FG tag copies do
  // (depth_capture::frame_generation_scope_live). A refusal while the scope is
  // silent changes nothing; while it is live it withdraws the FG source, as a
  // refused FG tag boundary does. The UI inputs are offered to the live UI
  // request as one exact batch (ui_mask::same_batch). A Streamline FG depth tag
  // of the frame DLSS-G evaluates (the viewport's newest, begun by its
  // constants), whose copy was recorded or waits for that frame's Present
  // while the scope's newest tag copy is younger than the source age, keeps
  // precedence (frame_generation::depth_tag_precedence), and so per kind does
  // a Streamline UI tag that recorded its copy within the source age: that
  // input adds no second copy (tag_copied). Every returned attempt must finish
  // after the original evaluation.
  frame_generation::attempt observe_frame_generation_inputs(const frame_generation::inputs &value);
  void finish_frame_generation_inputs(const frame_generation::attempt &value, bool successful);

  struct ui_observation_scope {
    std::uint64_t epoch{}, revision{}, sequence{}, tick{};
    std::uint32_t viewport{};
  };
  // A unique fresh viewport with a successful UI/HUD-less/color tag, independent
  // of FG mode or camera validity. This admits capture, never authorizes a mask
  // or claims correspondence to a ReShade presentation. UINT32_MAX searches all.
  bool query_ui_scope(std::uint32_t viewport, ui_observation_scope &output,
    std::uint32_t max_age_ms = sunshine_freshness::evidence_age_ms);

#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
  namespace versioning { struct number; }
  namespace testing {
    struct targets { void *constants{}, *tag{}, *tag_for_frame{}, *evaluate{}, *new_frame_token{}, *get_feature_function{}; };
    struct counters { std::uint64_t constants{}, tags{}, evaluations{}, dropped{}, invalid{}, vram_warnings{}, depth_untags{}; };
    // Real inline hooks on fixture functions, without pinning the process
    // executable, with the ABI family and buffer contract discovery selects
    // for an interposer whose version resource names exactly this version
    // (streamline_camera_version.h).
    bool install(const versioning::number &version, const targets &functions);
    counters counts();
    bool latest(std::uint32_t viewport, std::uint64_t &resource, bool &has_camera,
      bool &explicit_same_frame, bool &same_token_address);
    // The nomination tap, on a thread that armed it: nominate_depth_source
    // records the evaluation it received (without its retained sources) and,
    // as the call begins, its gates' answers and the depth they admit; the
    // call's finish_capture adds the SDK result. latest_nomination returns a
    // viewport's newest finished nomination of the current observation.
    struct nomination {
      evaluation_snapshot snapshot;
      bool successful{};
      bool path_selected{}, source_admissible{}; // source_path_selected, direct_source_admissible.
      std::array<bool, 3> tag_admissible{}; // direct_tag_admissible of each tag.
      // normalize_depth_frame of each tag that the two call gates and its own admit.
      std::array<sunshine_scene_depth::frame, 3> frames{};
      // Value translation only, never a post-return permission to read the
      // source: whether the call succeeded and the gates admitted snapshot.tags[index].
      bool nominates(unsigned index) const {
        return successful && path_selected && source_admissible && index < tag_admissible.size() && tag_admissible[index];
      }
    };
    void arm_nomination_tap(bool armed);
    bool latest_nomination(std::uint32_t viewport, nomination &out);
    // Fixture only: caller guarantees no callback can be running.
    void clear();
    // Exercise fail-closed observation loss without relying on scheduler luck.
    void lose_observation();
    bool discover_frame_generation_hooks();
    // Deterministic CPU-only contention fixture; lock/unlock on the same thread.
    void lock_records();
    void unlock_records();
    void lock_records_shared();
    void unlock_records_shared();
    void lock_tokens();
    void unlock_tokens();
    bool pin_tokens(unsigned index);
    void unpin_tokens();
    // The next count metadata reads fail, as a read that lost the race with
    // the writers on every attempt does (observer_snapshot.h); 0 ends it.
    void fail_metadata_reads(unsigned count);
    // Holds the DLSS-G evaluation supply lock exclusively until released, on
    // the calling thread; a hook that reads the supply must run on another.
    void hold_evaluation_supply();
    void release_evaluation_supply();
    // DLSS-G's evaluation supply: a copy of the scope at tick
    // (remember_evaluation_supply), and the one read of it a refused FG tag
    // boundary makes (evaluation_supplies).
    void supply_evaluation(std::uint64_t scope_epoch, std::uint32_t viewport, std::uint64_t tick);
    bool evaluation_supplied(std::uint64_t scope_epoch, std::uint32_t viewport);
    // The SL route of the viewport's evaluations (frame_generation_route).
    presentation_scope::fg_route route_of(std::uint32_t viewport);
    // Moves every enabled FG scope's enabled tick ms into the past.
    void age_frame_generation(std::uint64_t ms);
    // One explicitly armed fixture observation of the real pre-SDK policy.
    void arm_entry_policy();
    bool entry_policy(bool &selected, bool &admissible);
  }
#endif
}
