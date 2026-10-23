// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Why a depth input was refused, in the API contract matrix's terms (README,
// API contract matrix). The adapters name a production reason where they
// refuse: the Streamline probe's source and tag gates, the capture owner's
// record stage, NGX's parameters and DLSS-G's DLSSG.Depth outcome. classify()
// maps each to the REFUSE class the contract oracle
// (test_game3d_api_contract_oracle.h) gives the input that produces it, or to
// a class outside the contract. Only the TEST add-on records refusals, for
// SunshineDepthTestLastRefusal; the production add-on records and exports
// nothing, and no decision reads a class.
#include "frame_generation_inputs.h"
#include "streamline_depth_capture.h"

#include <cstdint>
#include <cstring>
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
  #include <cstdio>
  #include <Windows.h>
#endif

namespace sunshine_depth_refusal {
  // The oracle's REFUSE classes in its own order, then reasons outside them:
  // hold is the oracle's HOLD, not a refusal (an untag, S9; an SDK failure,
  // S21b; an open render pass, S23); absent means the call named no depth
  // input, or the game marked the frame as not rendering game frames;
  // internal is the add-on's own observation, frame tracking, identity or
  // capacity, never a contract decision; unclassified is a reason this file
  // does not know, which needs a row.
  enum class reason_class : std::uint32_t {
    none,
    state_unprovable,
    no_copy_point,
    lifetime,
    not_depth,
    malformed,
    unknown_abi,
    ambiguous_view,
    fg_scope,
    hold,
    absent,
    internal,
    unclassified
  };

  // The adapter that refused: a Streamline tag or evaluation, an NGX
  // evaluation, or DLSS-G's own NGX evaluation supplying a Streamline FG scope.
  enum class adapter : std::uint32_t {
    none,
    streamline,
    ngx,
    dlss_g
  };

  // What was judged: the observation, frame or scope the call belongs to
  // (source), the depth input it names (input: a Streamline tag, NGX's
  // parameters or DLSSG.Depth), or the capture owner's admission of its copy.
  enum class gate : std::uint32_t {
    none,
    source,
    input,
    capture
  };

  struct record {
    std::uint64_t serial {};  // Refusals recorded since the add-on loaded; 0 before the first.
    std::uint64_t epoch {}, sequence {};  // The adapter's observation epoch and call sequence; 0 when it assigned none.
    adapter provider {};
    gate where {};
    reason_class reason {};
    // The capture owner's decision when where is capture.
    sunshine_streamline::depth_capture::record_stage stage {};
    sunshine_streamline::depth_capture::status status {sunshine_streamline::depth_capture::status::unavailable};
    std::uint32_t viewport {UINT32_MAX};
    bool frame_generation {};  // A frame-generation input: a Streamline FG tag call or DLSS-G's evaluation.
    char name[48] {};  // The production reason: a probe gate, a record stage or an adapter outcome.
  };

  struct named_reason {
    const char *name;
    reason_class reason;
  };

  // The reasons the adapters name. The Streamline probe's gates
  // (streamline_camera_probe.cpp source_rejection, tag_rejection and
  // candidate_rejection) return literals, evidence_status names and
  // match_status names; NGX names its own.
  inline constexpr named_reason named_reasons[] {
    // Source gate literals.
    {"boundary_without_commands", reason_class::no_copy_point},  // A tag call with no command list (S2; S6 until R2 lands).
    {"no_observation", reason_class::internal},
    {"observation_lost", reason_class::internal},
    {"camera_reset", reason_class::internal},  // A reset without a usable camera (the add-on's camera policy).
    {"not_rendering_game_frames", reason_class::absent},
    {"metadata_busy", reason_class::internal},
    {"epoch_changed", reason_class::internal},
    {"frame_token_not_current", reason_class::internal},
    {"untracked_frame", reason_class::internal},
    // Evidence status names (name(evidence_status)).
    {"inactive", reason_class::internal},
    {"snapshot-busy", reason_class::internal},
    {"missing-evaluation", reason_class::internal},
    {"evaluation-failed", reason_class::hold},  // S21b.
    {"observation-lost", reason_class::internal},
    {"untracked-frame", reason_class::internal},
    {"missing-depth", reason_class::absent},
    {"unsupported-depth-tag", reason_class::malformed},
    {"selected-depth-not-ready", reason_class::internal},
    {"selected-source-mismatch", reason_class::internal},
    {"selected-extent-mismatch", reason_class::internal},
    {"stale-evaluation", reason_class::internal},  // The frame's camera is older than its tag.
    {"ambiguous-viewport", reason_class::ambiguous_view},  // S19.
    {"command-associated-evaluation-content-unverified", reason_class::internal},
    {"tracked-content-evaluation-coverage-incomplete", reason_class::internal},
    // Tag gate literals (streamline_camera_data.h tag_refusal and the gates after it).
    {"absent", reason_class::absent},
    {"unsupported_unreadable", reason_class::malformed},
    {"unsupported_header_version", reason_class::malformed},  // A tag or Resource header that is no known struct at version 1 or above (S14).
    {"unsupported_extension", reason_class::unknown_abi},  // An unknown chained struct (S14).
    {"unsupported_precision", reason_class::malformed},  // PrecisionInfo formula 2 or above, or scale 0 (S14).
    {"unsupported_resource_type", reason_class::not_depth},  // A declared ResourceType other than a texture (S12).
    {"unsupported_lifecycle", reason_class::malformed},  // Lifecycle 3 or more (S8).
    {"unsupported", reason_class::internal},  // A supported tag from a lost observation.
    {"not_depth", reason_class::not_depth},
    {"lifecycle_outlives_boundary", reason_class::fg_scope},  // A lifecycle 1 or 2 tag call outside frame generation (S20).
    {"constants_call_frame", reason_class::internal},
    {"no_retained_source", reason_class::malformed},  // A native the capture owner cannot retain as a single-sample 2D texture (D8).
    // Match status names (streamline_camera_data.cpp name(match_status)).
    {"missing-resource", reason_class::hold},  // A null native: an untag (S9).
    {"not-depth", reason_class::not_depth},
    {"wrong-viewport", reason_class::internal},
    {"stale-observation-epoch", reason_class::internal},
    {"wrong-frame", reason_class::lifetime},
    {"invalid-extent", reason_class::malformed},  // S16c.
    {"not-valid-until-present", reason_class::lifetime},  // An OnlyValidNow tag used after its call (S3).
    // The NGX evaluation adapter (ngx_depth_source.cpp).
    {"unregistered_feature", reason_class::internal},  // No observed or recoverable CreateFeature for the handle.
    {"missing_parameters", reason_class::malformed},  // Depth, its encoding or its subrect missing or invalid (N2, N5).
  };

  inline reason_class classify(const char *name) {
    if (!name) {
      return reason_class::unclassified;
    }
    for (const auto &entry : named_reasons) {
      if (!std::strcmp(entry.name, name)) {
        return entry.reason;
      }
    }
    return reason_class::unclassified;
  }

  // The capture owner's stage (streamline_depth_capture.cpp record_impl and
  // retain_source).
  constexpr reason_class classify(sunshine_streamline::depth_capture::record_stage stage) {
    using sunshine_streamline::depth_capture::record_stage;
    switch (stage) {
      case record_stage::not_attempted:
      case record_stage::recorded:
      case record_stage::retained:
        return reason_class::none;
      case record_stage::incomplete_state:  // D4.
      case record_stage::missing_state:  // D0, S10.
      case record_stage::conflicting_state:  // S11a.
      case record_stage::unsupported_state:  // D3b, D5b, X1, X2.
      case record_stage::unsupported_proof:  // S10.
        return reason_class::state_unprovable;
      case record_stage::unsupported_lifetime:
      case record_stage::source_lifetime:  // A Present since the tag (S8).
        return reason_class::lifetime;
      case record_stage::command_type:  // A list type the owner records no copy on.
      case record_stage::recording_closed:
        return reason_class::no_copy_point;
      case record_stage::recording_render_pass:  // S23.
        return reason_class::hold;
      case record_stage::malformed_input:
      case record_stage::missing_source:
      case record_stage::command_interface:
      case record_stage::resource_region:  // S16c.
      case record_stage::source_interface:
      case record_stage::source_description:  // D8.
        return reason_class::malformed;
      case record_stage::inactive:
      case record_stage::source_identity:
      case record_stage::observer_coverage:
      case record_stage::command_device:
      case record_stage::device_identity:
      case record_stage::recording_missing:
      case record_stage::recording_invalid:
      case record_stage::capacity:
      case record_stage::texture_allocation:
      case record_stage::descriptor_allocation:
      case record_stage::source_device:
      case record_stage::source_cookie:
        return reason_class::internal;
    }
    return reason_class::unclassified;
  }

  // DLSS-G's DLSSG.Depth outcome (frame_generation_inputs.h); a capture
  // refusal is classified by its stage.
  constexpr reason_class classify(sunshine_streamline::frame_generation::depth_outcome outcome, sunshine_streamline::depth_capture::record_stage stage) {
    using sunshine_streamline::frame_generation::depth_outcome;
    switch (outcome) {
      case depth_outcome::not_offered:
      case depth_outcome::copied:
      case depth_outcome::tag_copied:
        return reason_class::none;
      case depth_outcome::absent:
      case depth_outcome::null_resource:  // N4.
      case depth_outcome::subrect:  // N2.
        return reason_class::malformed;
      case depth_outcome::no_fg_scope:
        return reason_class::fg_scope;
      case depth_outcome::scope_changed:
      case depth_outcome::inactive:
        return reason_class::internal;
      case depth_outcome::capture:
        return classify(stage);
    }
    return reason_class::unclassified;
  }

  // The judged input of DLSS-G's outcome.
  constexpr gate gate_of(sunshine_streamline::frame_generation::depth_outcome outcome) {
    using sunshine_streamline::frame_generation::depth_outcome;
    switch (outcome) {
      case depth_outcome::absent:
      case depth_outcome::null_resource:
      case depth_outcome::subrect:
        return gate::input;
      case depth_outcome::no_fg_scope:
      case depth_outcome::scope_changed:
      case depth_outcome::inactive:
        return gate::source;
      case depth_outcome::capture:
        return gate::capture;
      default:
        return gate::none;
    }
  }

#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
  namespace detail {
    inline SRWLOCK lock = SRWLOCK_INIT;
    inline record latest;
  }  // namespace detail

  // The newest refusal, recorded where the adapter decides it and before any
  // log rate limit. A bounded POD copy under a lock no other code takes.
  inline void remember(record value, const char *name) {
    std::snprintf(value.name, sizeof(value.name), "%s", name ? name : "");
    AcquireSRWLockExclusive(&detail::lock);
    value.serial = detail::latest.serial + 1;
    detail::latest = value;
    ReleaseSRWLockExclusive(&detail::lock);
  }

  inline record latest() {
    AcquireSRWLockShared(&detail::lock);
    const auto value = detail::latest;
    ReleaseSRWLockShared(&detail::lock);
    return value;
  }
#endif
}  // namespace sunshine_depth_refusal
