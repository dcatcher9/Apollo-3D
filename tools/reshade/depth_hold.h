// SPDX-License-Identifier: GPL-3.0-only
#pragma once

// API lineage/storage and Generic content/anchor holds remain distinct.
// docs/reshade-sbs.md, depth_hold, owns their contract.
// The capture API header remains a transitive dependency through depth_addon.h
// and streamline_camera_probe.h; this move does not widen the include cut.
#include "depth_capture_types.h"
#include "presentation_scope.h"
#include "depth_addon.h"
#include "depth_capture_policy.h"
#include "freshness.h"
#include <algorithm>
#include <array>
#include <limits>

namespace sunshine_game3d::depth_output {
  using sunshine_streamline::depth_capture::acquisition_decision;
  using sunshine_streamline::depth_capture::packet;
  using sunshine_streamline::depth_capture::selection_policy;
  enum class display_action { copy_fresh, hold, invalidate };

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
      const bool fg = sunshine_scene_depth::is_fg_source(current.source_id, current.viewport);
      return streamline_lineage(before, current.epoch, current.viewport) && current.provider == before.provider &&
        fg != before.frame_generation_input;
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
      // Never older depth after newer (owner ruling 10-30 Q3, C8): a
      // candidate of the retained copy's display lineage (its viewport's
      // Streamline FG input or ordinary source of its epoch) published before
      // the retained copy (capture ids follow publication order) does not
      // replace it. The retained copy is held under the rules below until a
      // copy published after it is readable, or it expires. Until 10-31 FG
      // Off showed such a copy whenever selection picked it: seed
      // c40a42f921da5ba0 at frame 149 showed frame 144's SR copy after frame
      // 146's FG copy and held it for about 125 ms.
      const bool older = current.source_selected && current.capture_id && previous.capture_id &&
        current.capture_id < previous.capture_id && current.provider == previous.metadata.provider &&
        streamline_lineage(previous.metadata, current.epoch, current.viewport);
      if (current.source_selected && readable && !older) {
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
      // An older candidate of the lineage (above) is no newer identity: it
      // contradicts nothing, and its storage and encoding, never shown, end
      // nothing.
      const bool other_source = !older && !same_source(previous, current);
      if (other_source && !lineage_successor(previous, current)) {
        out.reason = "source_changed";
        return out;
      }
      if (!other_source && !older && current.source_selected && !same_storage(previous, candidate)) {
        out.reason = "depth_format_changed";
        return out;
      }
      if (!other_source && !older && current.source_selected && before.projection.encoding != candidate.metadata.projection.encoding) {
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
      out.reason = older ? "retained_newer_copy" : !other_source ? "retained_copy" :
        before.frame_generation_input ? "FG_successor_pending" : "FG_copy_pending";
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
  // only its own lineage's copy. A selected copy of the retained copy's lineage
  // published before it never replaces it (retained_newer_copy): no older
  // depth after newer. hold still requires the cache owner's local
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

  struct display_cache_context {
    std::uint32_t color_width{}, color_height{};
    bool immediate_commands{};
    std::uint64_t queue{}, texture{}, view{}, now{};
  };

  // One successful private display copy. Source admission belongs to capture;
  // this owner only checks whether its presentation storage can still be used.
  // It owns no game resource or capture lease and performs no runtime/GPU calls.
  class display_depth_cache {
  public:
    const sunshine_depth::frame_depth &depth() const { return depth_; }
    bool matches_color(std::uint32_t width, std::uint32_t height) const {
      return width == color_width_ && height == color_height_;
    }

    retained_depth reference() const {
      if (!depth_.ready) return {};
      retained_depth out;
      out.metadata = depth_.provided;
      out.capture_id = capture_id_;
      out.real_present = depth_.frame_index;
      out.present_stamp = present_stamp_;
      out.width = depth_.width;
      out.height = depth_.height;
      out.format = format_;
      out.view_format = view_format_;
      out.area = {depth_.x, depth_.y, depth_.active_width, depth_.active_height};
      return out;
    }

    // Only the caller's successful fresh copy can arm the cache. In particular,
    // feeding a held output back here cannot refresh its age or NGX present limit.
    // format and view_format: the copy's storage family and the view it is
    // read through (packet::format, packet::srv_format).
    // present_stamp: the copy's frame's present ordinal (packet::present).
    void commit(const sunshine_depth::frame_depth &value, std::uint64_t capture_id,
        std::uint32_t format, std::uint32_t view_format, std::uint32_t color_width, std::uint32_t color_height,
        std::uint64_t present_stamp = 0) {
      if (!value.ready || value.reused_depth || !capture_id) {
        invalidate("invalid_fresh_copy");
        return;
      }
      depth_ = value;
      capture_id_ = capture_id;
      present_stamp_ = present_stamp;
      format_ = format;
      view_format_ = view_format;
      color_width_ = color_width;
      color_height_ = color_height;
      reason_ = "none";
    }

    // The reason must remain valid until the next commit/invalidation; production
    // callers use static diagnostic strings. Invalidated pixels cannot revive.
    void invalidate(const char *reason) {
      depth_ = {};
      capture_id_ = present_stamp_ = 0;
      format_ = view_format_ = color_width_ = color_height_ = 0;
      reason_ = reason ? reason : "invalidated";
    }
    const char *reason() const { return reason_; }

    bool reuse(const display_decision &decision, const display_cache_context &context,
        std::uint64_t present, sunshine_depth::frame_depth &out) {
      out = {};
      if (decision.action != display_action::hold) {
        invalidate(decision.reason);
        return false;
      }
      if (!depth_.ready) return false;
      if (!matches_color(context.color_width, context.color_height))
        return reject("color_shape_changed");
      if (!context.immediate_commands) return reject("consumer_command_changed");
      if (!context.queue || context.queue != depth_.command_queue)
        return reject("consumer_queue_changed");
      if (!context.texture || !context.view || context.texture != depth_.resource.handle ||
          context.view != depth_.shader_resource.handle)
        return reject("display_storage_changed");
      // decide_display's own_frame: the frame this Present still presents.
      if (!decision.own_frame && !sunshine_freshness::within_source_age(depth_.provided.tick, context.now))
        return reject("retained_depth_expired");
      out = depth_;
      out.frame_index = present;
      out.reused_depth = true;
      return true;
    }

  private:
    bool reject(const char *reason) { invalidate(reason); return false; }
    sunshine_depth::frame_depth depth_;
    std::uint64_t capture_id_{}, present_stamp_{};
    std::uint32_t format_{}, view_format_{}, color_width_{}, color_height_{};
    const char *reason_{"no_retained_depth"};
  };

// Whether a Present may hold the shown capture (judge_hold_gate): open, or
// refused because the anchor's set was cleared without drawing (cleared) or a
// scene pass started in another depth-stencil (cut).
enum class hold_gate { open, cleared, cut };
// The hold gates (docs/reshade-sbs.md, Selecting scene depth), judged on a
// Present that advanced the depth frame from its snapshot, before any hold:
// - cleared: the anchor's capture set (with the held source) was cleared or
//   discarded and none of it worked: the game shows a screen without its scene
//   (a pause, inventory, loading screen or video that keeps clearing depth);
// - cut: a shape-eligible depth-stencil outside that set drew although it had
//   no work within the selector's activity grace (`grace` depth frames before
//   this one, as long as a candidate still counts as active) and does not
//   draw at the set's cadence: a scene pass that starts now in another buffer
//   (a cutscene, photo mode or 3D menu), so the held depth would be another
//   camera's.
// Either way the Present holds nothing: it is mono. A set left untouched keeps
// its hold (a frame presented twice, frame generation, a colour-only pass, the
// Presents an every-other-Present source skips), and so does one that worked
// but whose copy was refused or failed (a missing frame). A source that keeps
// drawing (a HUD with its own depth-stencil, an every-Present lower-resolution
// scene, a member of an A/B or A/B/C rotation on its turn) is no cut, and so
// is one that draws at the set's cadence: a member of a rotation the set's own
// members show, on its turn, before it qualifies into the set (A/B/C/D, whose
// turns come four depth frames apart). It draws at that cadence when its idle
// interval equals the interval between the last two worked depth frames of a
// set member whose next turn is still to come: a stopped set lends none, and
// a single long gap (a member resuming its scene) matches nothing else by
// chance. Such a member's turn is a slot no set member owns, so a set member
// whose turn this frame is (one interval after its last work) and that did not
// work lends no cadence to anyone: a buffer drawing in its place took its
// turn, not a free slot, and is a cut even when it last drew exactly one
// cadence before. No knob of its own.
template<class Snapshot, class MatchesShape>
inline hold_gate judge_hold_gate(const Snapshot &snapshot,
	const sunshine_depth_capture::capture_set &captures, uint64_t held, uint64_t frame,
	uint32_t color_width, uint32_t color_height, bool resolve_depth, uint64_t grace, MatchesShape &&candidate_matches_shape)
{
	constexpr auto none = std::numeric_limits<uint64_t>::max();
	const auto in_set = [&](uint64_t id) {
		if (id == held) return true;
		for (unsigned i = 0; i != captures.count; ++i)
			if (captures.records[i].identity == id) return true;
		return false;
	};
	// The set's cadence (above): at most one interval per member and the held
	// source, from the members whose next turn is still to come; `skipped`: a
	// member whose turn this frame is did not work.
	std::array<uint64_t, sunshine_depth_capture::maximum_sources + 1> cadence {};
	unsigned cadences = 0;
	bool erased = false, worked = false, skipped = false;
	for (const auto &[value, info] : snapshot)
	{
		if (!in_set(info.identity)) continue;
		const auto &stats = info.last_frame_stats;
		if (info.prior_work_frame != none && stats.total.has_work() && info.last_used_in_frame <= frame)
		{
			const auto interval = info.last_used_in_frame - info.prior_work_frame, since = frame - info.last_used_in_frame;
			if (since < interval && cadences != cadence.size()) cadence[cadences++] = interval;
			else if (since == interval) skipped = true;
		}
		if (info.last_used_in_frame != frame) continue;
		erased = erased || stats.depth_erased;
		worked = worked || stats.total.has_work();
	}
	if (erased && !worked) return hold_gate::cleared;
	for (const auto &[value, info] : snapshot)
	{
		if (info.last_used_in_frame != frame || in_set(info.identity) || !info.last_frame_stats.total.drew() ||
			!(info.desc.texture.samples <= 1 || resolve_depth) || !candidate_matches_shape(info, color_width, color_height)) continue;
		const auto idle = info.prior_work_frame == none ? none : frame - info.prior_work_frame;
		if (idle > grace && (skipped || std::find(cadence.begin(), cadence.begin() + cadences, idle) == cadence.begin() + cadences))
			return hold_gate::cut;
	}
	return hold_gate::open;
}

// The value-only half of Generic retention. Clock evaluation remains lazy at
// the original final age clause; resource-map/view validation stays in its TU.
template<class Data, class Clock>
inline bool generic_capture_matches(const Data &data, uint32_t color_width,
    uint32_t color_height, bool check_age, Clock &&now) {
  const auto &shown = data.shown;
  const auto &record = shown.record;
  return !(data.hold_verdict != hold_gate::open ||
		!record.identity || record.direct || !shown.content_serial || !data.using_backup_texture ||
		record.runtime != data.runtime_epoch || shown.anchor != data.anchor_identity ||
		record.identity != data.selected_identity || record.source != data.selected_depth_stencil.handle ||
		record.view != data.selected_shader_resource.handle ||
		shown.color_width != color_width || shown.color_height != color_height ||
		(check_age && !sunshine_freshness::within_source_age(shown.tick, now())));
}
}
