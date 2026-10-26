// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "freshness.h"
#include "streamline_depth_capture.h"

namespace sunshine_streamline::depth_capture {
  enum class display_action { copy_fresh, hold, invalidate };

  // Description of the last successful private display copy. It retains no
  // game-resource lease, capture-pool allocation, or independently aged copy.
  // format is the copy's storage family, view_format the view it is read
  // through (packet::format and packet::srv_format).
  struct retained_depth {
    sunshine_scene_depth::frame metadata;
    std::uint64_t capture_id{}, real_present{};
    std::uint32_t width{}, height{}, format{}, view_format{};
    sunshine_scene_depth::extent area;
  };
  struct display_decision {
    display_action action{display_action::invalidate};
    const char *reason{"no_retained_depth"};
  };

  namespace detail {
    // The retained copy reads like the candidate: one storage family through
    // one view, of one resource kind. Its allocation size and active rectangle
    // are not compared: a dynamic render size gives the same source a new
    // allocation or extent every few frames, and the held copy is presented
    // with its own size and rectangle (consumers map that rectangle).
    inline bool same_storage(const retained_depth &previous, const packet &next) {
      return next.format == previous.format && next.srv_format == previous.view_format &&
        next.metadata.resource.kind == previous.metadata.resource.kind;
    }
    // Logical source identity survives ordinary rotation of resource pointers.
    // Without any current nomination there is no newer identity to contradict.
    // Required FG without a nomination names its scope but no sequence (zero):
    // a different scope still contradicts, a missing sequence does not.
    inline bool same_source(const retained_depth &previous, const acquisition_decision &current) {
      const auto &before = previous.metadata;
      if (!current.epoch) return true;
      return current.provider == before.provider && current.epoch == before.epoch &&
        current.source_id == before.source_id && current.viewport == before.viewport &&
        (!current.sequence || current.sequence >= before.sequence);
    }
  }

  namespace detail {
    // decide_display, for a selected candidate whose pixels can be copied at
    // this Present (readable) or no longer can (its copy failed).
    inline display_decision decide(const acquisition_decision &current, const packet &candidate,
        const retained_depth &previous, selection_policy policy, std::uint64_t now, bool readable) {
      display_decision out;
      if (policy.exclude_unconfirmed_fg && current.source_selected && candidate.metadata.frame_generation_input) {
        out.reason = "FG_scope_unconfirmed";
        return out;
      }
      if (current.source_selected && readable) {
        out.action = display_action::copy_fresh;
        out.reason = "fresh_copy_candidate";
        return out;
      }
      if (!previous.capture_id || !previous.metadata.epoch) return out;
      const auto &before = previous.metadata;
      if (policy.exclude_unconfirmed_fg && before.frame_generation_input) {
        out.reason = "FG_scope_unconfirmed";
        return out;
      }
      // FG owns one logical SL input; ordinary depth cannot stand in for it.
      if (policy.require_frame_generation && (!before.frame_generation_input ||
          before.provider != sunshine_scene_depth::provider_kind::streamline ||
          before.epoch != policy.epoch || before.viewport != policy.viewport)) {
        out.reason = "FG_scope_changed";
        return out;
      }
      if (!same_source(previous, current)) {
        out.reason = "source_changed";
        return out;
      }
      if (current.source_selected && !same_storage(previous, candidate)) {
        out.reason = "depth_format_changed";
        return out;
      }
      if (current.source_selected && before.projection.encoding != candidate.metadata.projection.encoding) {
        out.reason = "depth_encoding_changed";
        return out;
      }
      if (!sunshine_freshness::within_source_age(before.tick, now)) {
        out.reason = "retained_depth_expired";
        return out;
      }
      out.action = display_action::hold;
      out.reason = "retained_copy";
      return out;
    }
  }

  // Source policy only. copy_fresh authorizes an attempt, never a successful
  // copy (decide_failed_copy decides a failed one). Otherwise the newest
  // successful private copy is held while it is younger than the source age
  // (freshness.h) and the provider still names the same source.
  // A pending, failed or missing newer frame, a new render size of the same
  // source, a camera reset and tag or observation bookkeeping never discard
  // it; a changed source, storage format, view or resource kind, encoding and
  // its age do. hold still requires the cache owner's local
  // storage, color extent, immediate-command and consumer-queue checks. Only a
  // successful fresh copy may replace retained_depth; holds never refresh it.
  inline display_decision decide_display(const acquisition_decision &current, const packet &candidate,
      const retained_depth &previous, selection_policy policy, std::uint64_t now) {
    return detail::decide(current, candidate, previous, policy, now, candidate.pixel_ready);
  }

  // A copy_fresh attempt that failed. The consumer records its copy into the
  // display storage only once every check passed (copy_current), so a failed
  // attempt leaves the newest successful private copy as it was, and the
  // candidate is a failed newer frame of its source: the retained copy is
  // held by decide_display's rules for a pending one (the same source,
  // storage, encoding and FG scope, younger than the source age:
  // retained_after_failed_copy). A failure refuses the candidate's pixels
  // only. An observation loss, for one (a hook installed for a queue created
  // mid-session), invalidates the recordings open when it happened, the
  // consumer's among them (recording_invalid), never a copy already finished
  // in the display storage on the presenting queue. Where those rules hold
  // nothing, the Present is mono (display_copy_failed).
  inline display_decision decide_failed_copy(const acquisition_decision &current, const packet &candidate,
      const retained_depth &previous, selection_policy policy, std::uint64_t now) {
    auto out = detail::decide(current, candidate, previous, policy, now, false);
    out.reason = out.action == display_action::hold ? "retained_after_failed_copy" : "display_copy_failed";
    return out;
  }

  // FG keeps its viewport while it can supply. An enabled FG scope that is
  // silent at this Present (its newest frame had no FG copy) lets the same
  // viewport's ordinary SR/RR depth stand in only once its own newest copy
  // can no longer be held: while the display holds a copy of that scope made
  // since FG On (policy.enabled_tick), younger than the source age, that copy
  // is held over a fresh or pending ordinary candidate. A frame without an FG
  // copy is then held as any missing frame of a source is, instead of
  // switching the viewport's source (which restarts the placement
  // calibration) to SR and back. policy: the confirmed FG policy, not the
  // silent scope's selection.
  inline bool frame_generation_holds(const retained_depth &previous, selection_policy policy, std::uint64_t now) {
    const auto &before = previous.metadata;
    return policy.require_frame_generation && policy.epoch && previous.capture_id && before.frame_generation_input &&
      before.provider == sunshine_scene_depth::provider_kind::streamline && before.epoch == policy.epoch &&
      before.viewport == policy.viewport && before.tick >= policy.enabled_tick &&
      sunshine_freshness::within_source_age(before.tick, now);
  }

  // An FG copy still in flight has not yet earned its viewport. While an
  // enabled FG scope is silent and the acquisition names only that scope's
  // own capture, without readable pixels (Streamline resets DLSS-G's compute
  // recording frames after the copy, and DLSS-G's nomination replaces the
  // viewport's SR observation), the display holds the viewport's ordinary
  // copy of the same epoch, younger than the source age, instead of dropping
  // it as a changed source. The scope turns live once that FG copy is
  // readable (frame_generation_scope_live) and takes the viewport then.
  // The capture may be selected (its nomination's authority) with its pixels
  // pending (candidate.pixel_ready). policy: the confirmed FG policy, not the
  // silent scope's selection.
  inline bool frame_generation_pending(const retained_depth &previous, const acquisition_decision &current,
      const packet &candidate, selection_policy policy, std::uint64_t now) {
    const auto &before = previous.metadata;
    return policy.require_frame_generation && policy.epoch && !(current.source_selected && candidate.pixel_ready) &&
      current.provider == sunshine_scene_depth::provider_kind::streamline && current.epoch == policy.epoch &&
      current.viewport == policy.viewport &&
      current.source_id == sunshine_scene_depth::encode_source_id(sunshine_scene_depth::fg_depth_key(policy.viewport)) &&
      previous.capture_id && !before.frame_generation_input &&
      before.provider == sunshine_scene_depth::provider_kind::streamline && before.epoch == policy.epoch &&
      before.viewport == policy.viewport && sunshine_freshness::within_source_age(before.tick, now);
  }
}
