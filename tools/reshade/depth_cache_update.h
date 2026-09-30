// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "streamline_depth_capture.h"

namespace sunshine_streamline::depth_capture {
  enum class display_action { copy_fresh, hold, invalidate };

  // Description of the last successful private display copy. It retains no
  // game-resource lease, capture-pool allocation, or independently aged copy.
  struct retained_depth {
    sunshine_scene_depth::frame metadata;
    std::uint64_t capture_id{}, real_present{};
    std::uint32_t width{}, height{}, format{};
    sunshine_scene_depth::extent area;
  };
  struct display_decision {
    display_action action{display_action::invalidate};
    const char *reason{"no_retained_depth"};
  };

  namespace detail {
    inline bool same_layout(const retained_depth &previous, const packet &next) {
      return next.width == previous.width && next.height == previous.height && next.format == previous.format &&
        next.area.left == previous.area.left && next.area.top == previous.area.top &&
        next.area.width == previous.area.width && next.area.height == previous.area.height;
    }
    // Logical source identity survives ordinary rotation of resource pointers.
    // Without any current nomination there is no newer identity to contradict.
    inline bool same_source(const retained_depth &previous, const acquisition_decision &current) {
      const auto &before = previous.metadata;
      if (!current.epoch) return true;
      return current.provider == before.provider && current.epoch == before.epoch &&
        current.source_id == before.source_id && current.viewport == before.viewport &&
        current.sequence >= before.sequence;
    }
  }

  // Source policy only. copy_fresh authorizes an attempt, never a successful
  // copy: failure must invalidate, not re-enter this policy to hold old pixels.
  // Otherwise the newest successful private copy is held while it is younger
  // than maximum_source_age_ms and the provider still names the same source.
  // A pending, failed or missing newer frame, a camera reset and tag or
  // observation bookkeeping never discard it; a changed source, layout or
  // encoding and its age do. hold still requires the cache owner's local
  // storage, color extent, immediate-command and consumer-queue checks. Only a
  // successful fresh copy may replace retained_depth; holds never refresh it.
  inline display_decision decide_display(const acquisition_decision &current, const packet &candidate,
      const retained_depth &previous, selection_policy policy, std::uint64_t now) {
    display_decision out;
    if (policy.exclude_unconfirmed_fg && current.source_selected && candidate.metadata.frame_generation_input) {
      out.reason = "FG_scope_unconfirmed";
      return out;
    }
    if (current.source_selected && (candidate.shared_preservation || candidate.pixel_ready)) {
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
    if (!detail::same_source(previous, current)) {
      out.reason = "source_changed";
      return out;
    }
    if (current.source_selected && !detail::same_layout(previous, candidate)) {
      out.reason = "depth_shape_changed";
      return out;
    }
    if (current.source_selected && before.projection.encoding != candidate.metadata.projection.encoding) {
      out.reason = "depth_encoding_changed";
      return out;
    }
    if (!before.tick || now < before.tick || now - before.tick >= sunshine_scene_depth::maximum_source_age_ms) {
      out.reason = "retained_depth_expired";
      return out;
    }
    out.action = display_action::hold;
    out.reason = "retained_copy";
    return out;
  }
}
