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
    // The present ordinal of the frame the copy is (packet::present),
    // compared with acquisition_decision::presenting_floor.
    std::uint64_t present_stamp{};
  };
  struct display_decision {
    display_action action{display_action::invalidate};
    const char *reason{"no_retained_depth"};
    // A hold of the newest evaluated frame's own copy at a Present that
    // still presents that frame: no source age applies (display_depth_cache
    // ::reuse).
    bool own_frame{};
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
    // Without any current nomination there is no newer identity to contradict
    // (a head refused before its capture existed names its view's,
    // classify_acquisition).
    // Required FG without a nomination names its scope but no sequence (zero):
    // a different scope still contradicts, a missing sequence does not.
    inline bool same_source(const retained_depth &previous, const acquisition_decision &current) {
      const auto &before = previous.metadata;
      if (!current.epoch) return true;
      return current.provider == before.provider && current.epoch == before.epoch &&
        current.source_id == before.source_id && current.viewport == before.viewport &&
        (!current.sequence || current.sequence >= before.sequence);
    }
    // A Streamline copy of a viewport's display lineage: its FG input (FG
    // copies are Streamline's) or its ordinary SR/RR source, of one epoch.
    inline bool streamline_lineage(const sunshine_scene_depth::frame &copy, std::uint64_t epoch, std::uint32_t viewport) {
      return copy.provider == sunshine_scene_depth::provider_kind::streamline && copy.epoch == epoch &&
        copy.viewport == viewport;
    }
    // The other source of the retained copy's lineage: the acquisition names
    // the viewport's ordinary source after an FG copy (FG Off, or a silent
    // scope) or its FG input after an ordinary copy (FG On, or a scope that
    // supplies again). It takes the viewport at its first readable frame,
    // and until then the retained copy is held as any source's missing
    // frame is, in either direction.
    inline bool lineage_successor(const retained_depth &previous, const acquisition_decision &current) {
      const auto &before = previous.metadata;
      const bool fg = current.source_id ==
        sunshine_scene_depth::encode_source_id(sunshine_scene_depth::fg_depth_key(current.viewport));
      return streamline_lineage(before, current.epoch, current.viewport) && current.provider == before.provider &&
        fg != before.frame_generation_input;
    }
  }

  // A retained Streamline FG copy younger than the source age: what
  // frame_generation_holds and frame_generation_retained keep.
  inline bool frame_generation_copy(const retained_depth &previous, std::uint64_t now) {
    const auto &before = previous.metadata;
    return previous.capture_id && before.frame_generation_input &&
      before.provider == sunshine_scene_depth::provider_kind::streamline &&
      sunshine_freshness::within_source_age(before.tick, now);
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
      // FG owns one logical SL input: only its own lineage's copy (its FG
      // input's, or the viewport's ordinary copy it follows) stands in for it.
      if (policy.require_frame_generation && !streamline_lineage(before, policy.epoch, policy.viewport)) {
        out.reason = "FG_scope_changed";
        return out;
      }
      // Another source ends the hold, except the other source of the
      // retained copy's lineage (lineage_successor): it takes the viewport at
      // its first readable frame (copied above), and until then the copy is
      // held as any source's missing frame is. Storage and encoding compare
      // frames of one source only.
      const bool other_source = !same_source(previous, current);
      if (other_source && !lineage_successor(previous, current)) {
        out.reason = "source_changed";
        return out;
      }
      if (!other_source && current.source_selected && !same_storage(previous, candidate)) {
        out.reason = "depth_format_changed";
        return out;
      }
      if (!other_source && current.source_selected && before.projection.encoding != candidate.metadata.projection.encoding) {
        out.reason = "depth_encoding_changed";
        return out;
      }
      // The retained copy is the frame this Present still presents (FG On:
      // its generated and real Presents in either order, frame_presenting):
      // its own depth, however long the frame took since its evaluation.
      // Every other hold keeps the source age (until the 10-27 review a hitch
      // between a frame's evaluation and its last Present made that Present
      // mono, retained_depth_expired).
      out.own_frame = previous.present_stamp && current.presenting_floor && previous.present_stamp >= current.presenting_floor;
      if (!out.own_frame && !sunshine_freshness::within_source_age(before.tick, now)) {
        out.reason = "retained_depth_expired";
        return out;
      }
      out.action = display_action::hold;
      out.reason = !other_source ? "retained_copy" : before.frame_generation_input ? "FG_successor_pending" : "FG_copy_pending";
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
  // its age do. A viewport's Streamline FG input and its ordinary SR/RR
  // source of one epoch are one display lineage (detail::lineage_successor):
  // the other source of the retained copy's lineage is no changed source
  // until its first readable frame, and the copy is held meanwhile, an FG
  // copy for the ordinary source after FG Off or in a silent scope
  // (FG_successor_pending), as for an SR copy on another queue, readable
  // only once its recording retired, and an ordinary copy for the FG input
  // after FG On or in a silent scope (FG_copy_pending), as for DLSS-G's
  // compute copy, readable only frames later. A required FG scope accepts
  // only its own lineage's copy. hold still requires the cache owner's local
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
    return policy.require_frame_generation && policy.epoch && frame_generation_copy(previous, now) &&
      before.epoch == policy.epoch && before.viewport == policy.viewport && before.tick >= policy.enabled_tick;
  }

  // FG Off retires the FG source, which releases its queue's API association
  // (retire_source), while the display may still hold that scope's newest
  // copy. Within the source age it is held as any source's missing frame is:
  // while the display holds an FG copy younger than the source age and FG is
  // not revoked (an unconfirmed scope refuses its copies), the queue stays
  // the API's, so decide_display holds the copy until it expires, a readable
  // frame of another source replaces it or a source outside its lineage
  // drops it. Only then may the queue hand over to Generic. The successor's
  // copies recorded since the shown FG copy stay admissible at FG Off
  // (retire_source retires only the FG role), so the first of them readable
  // replaces the held copy at once.
  inline bool frame_generation_retained(const retained_depth &previous, selection_policy policy, std::uint64_t now) {
    return !policy.exclude_unconfirmed_fg && frame_generation_copy(previous, now);
  }
}
