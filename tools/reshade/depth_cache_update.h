// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "streamline_depth_capture.h"

namespace sunshine_streamline::depth_capture {
  enum class display_action { copy_fresh, hold, invalidate };
  enum class hold_kind { none, frame_generation, ngx_pending };

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
    hold_kind hold{};
    std::uint64_t current_check_revision{}, retained_check_revision{};
    int current_observation{-1};
  };

  namespace detail {
    inline bool same_fg_source(const retained_depth &previous, const acquisition_decision &current) {
      const auto &before = previous.metadata;
      // Logical source identity survives ordinary rotation of resource pointers.
      return previous.capture_id && before.frame_generation_input && before.epoch && before.source_id &&
        before.provider == sunshine_scene_depth::provider_kind::streamline &&
        current.provider == before.provider && current.epoch == before.epoch &&
        current.source_id == before.source_id && current.viewport == before.viewport &&
        current.sequence >= before.sequence && current.source_valid;
    }
    inline bool same_layout(const retained_depth &previous, const packet &next) {
      return next.width == previous.width && next.height == previous.height && next.format == previous.format &&
        next.area.left == previous.area.left && next.area.top == previous.area.top &&
        next.area.width == previous.area.width && next.area.height == previous.area.height;
    }
    inline bool ngx_predecessor(const retained_depth &previous, const packet &pending,
        const acquisition_decision &current, std::uint64_t present, std::uint64_t now) {
      const auto &before = previous.metadata;
      const auto &next = pending.metadata;
      const auto &a = before.projection;
      const auto &b = next.projection;
      // Capture already proved the submitted successor and exact consumed
      // predecessor. Bind that proof to this private copy for one presentation.
      return previous.capture_id && previous.real_present && present > previous.real_present &&
        present - previous.real_present == 1 && before.tick && now >= before.tick &&
        now - before.tick < sunshine_scene_depth::maximum_source_age_ms &&
        before.provider == sunshine_scene_depth::provider_kind::ngx && !before.frame_generation_input &&
        next.provider == before.provider && !next.frame_generation_input &&
        before.epoch && before.source_id && before.sequence && next.epoch == before.epoch &&
        next.source_id == before.source_id && next.viewport == before.viewport && next.sequence > before.sequence &&
        next.tick >= before.tick && now >= next.tick &&
        next.observation_revision == before.observation_revision &&
        before.feedback.revision && next.feedback.revision == before.feedback.revision &&
        !before.feedback.reset && !next.feedback.reset &&
        a.supplied == b.supplied && a.direction_supplied == b.direction_supplied && a.reversed == b.reversed &&
        a.depth_offset == b.depth_offset && a.depth_scale == b.depth_scale && a.raw_scale == b.raw_scale && a.raw_bias == b.raw_bias &&
        same_layout(previous, pending) && next.resource.kind == before.resource.kind &&
        !pending.shared_preservation && !pending.pixel_ready && pending.capture_id &&
        current.source_selected && current.source_valid &&
        current.pending_ngx_previous_capture == previous.capture_id && current.capture_id == pending.capture_id &&
        current.provider == next.provider && current.epoch == next.epoch && current.source_id == next.source_id &&
        current.sequence == next.sequence && current.viewport == next.viewport;
    }
  }

  // Source policy only. copy_fresh authorizes an attempt, never a successful
  // copy: failure must invalidate, not re-enter this policy to hold old pixels.
  // hold still requires the cache owner's existing local storage, color extent,
  // immediate-command and consumer-queue checks. Only a successful fresh copy
  // may replace retained_depth; holds never refresh its timestamp or identity.
  inline display_decision decide_display(const acquisition_decision &current, const packet &candidate,
      const retained_depth &previous, selection_policy policy, std::uint64_t present,
      std::uint64_t now, std::uint64_t observed_sl_revision) {
    display_decision out;
    const bool check_current = current.source_selected &&
      candidate.metadata.provider == sunshine_scene_depth::provider_kind::streamline;
    if (check_current) out.current_check_revision = observed_sl_revision;
    out.current_observation = !check_current || candidate.metadata.observation_revision == observed_sl_revision;
    if (current.source_selected && out.current_observation && (candidate.shared_preservation || candidate.pixel_ready)) {
      out.action = display_action::copy_fresh;
      out.reason = "fresh_copy_candidate";
      return out;
    }
    if (!previous.capture_id) return out;

    out.reason = "reuse_not_authorized";
    const bool ngx_pending = current.source_selected && !policy.require_frame_generation &&
      detail::ngx_predecessor(previous, candidate, current, present, now);
    if (!ngx_pending && !detail::same_fg_source(previous, current)) return out;
    const bool check_retained = previous.metadata.provider == sunshine_scene_depth::provider_kind::streamline;
    if (check_retained) out.retained_check_revision = observed_sl_revision;
    if (!ngx_pending) {
      const bool repeated = !current.source_selected && current.repeated_frame &&
        current.capture_id == previous.capture_id && current.sequence == previous.metadata.sequence;
      const bool pending = current.pending_frame &&
        (!current.source_selected || (!candidate.shared_preservation && !candidate.pixel_ready));
      const bool enabled = policy.require_frame_generation && policy.epoch == previous.metadata.epoch &&
        policy.viewport == previous.metadata.viewport;
      if (!(repeated || pending) || !enabled) return out;
    }
    if (current.source_selected && !detail::same_layout(previous, candidate)) {
      out.reason = "depth_shape_changed";
      return out;
    }
    if (check_retained && previous.metadata.observation_revision != observed_sl_revision) {
      out.reason = "retained_observation_changed";
      return out;
    }
    if (!previous.metadata.tick || now < previous.metadata.tick ||
        now - previous.metadata.tick >= sunshine_scene_depth::maximum_source_age_ms) {
      out.reason = "retained_depth_expired";
      return out;
    }
    out.action = display_action::hold;
    out.hold = ngx_pending ? hold_kind::ngx_pending : hold_kind::frame_generation;
    out.reason = "retained_copy";
    return out;
  }
}
