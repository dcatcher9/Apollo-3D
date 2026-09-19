// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "camera_sample_binding.h"
#include "streamline_camera_probe.h"
#include <cmath>

namespace sunshine_camera_binding {
  inline depth_copy_key observed_copy_key(const sunshine_streamline::content::copy_snapshot &copy) noexcept {
    return {copy.copy_id, copy.ledger_epoch, copy.recording.command, copy.recording.recording_generation,
      copy.source.version, copy.backup_registration};
  }
  inline bool same_copy_key(const depth_copy_key &a, const depth_copy_key &b) noexcept {
    return a.copy_id == b.copy_id && a.ledger_epoch == b.ledger_epoch &&
      a.recording_command == b.recording_command && a.recording_generation == b.recording_generation &&
      a.source_content_version == b.source_content_version && a.backup_registration == b.backup_registration;
  }
  inline bool same_marker(const sunshine_streamline::commands::recording_marker &a,
      const sunshine_streamline::commands::recording_marker &b) noexcept {
    return a.command == b.command && a.object_generation == b.object_generation &&
      a.recording_generation == b.recording_generation && a.event == b.event && a.epoch == b.epoch && a.loss == b.loss;
  }
  inline bool same_selection_observation(const sunshine_streamline::selected_depth &a,
      const sunshine_streamline::selected_depth &b) noexcept {
    return a.resource == b.resource && a.source_id == b.source_id && a.layout_epoch == b.layout_epoch &&
      a.frame_index == b.frame_index && a.width == b.width && a.height == b.height && a.x == b.x && a.y == b.y &&
      a.active_width == b.active_width && a.active_height == b.active_height && a.ready == b.ready &&
      a.backup_resource == b.backup_resource && a.backup_id == b.backup_id && a.command_queue == b.command_queue;
  }
  // Called once at GPU submission. This copies a weak observation, never a
  // registered camera. No latest probe state may be attached on completion.
  inline void capture_camera_observation(submission &target, const sunshine_streamline::selected_depth &selected,
      const sunshine_streamline::evaluation_snapshot &evidence) noexcept {
    namespace sl = sunshine_streamline;
    target.projection_associated = false;
    target.camera.proof_admitted = false;
    target.camera.A = target.camera.B = 0;
    auto &out = target.observation;
    out = {};
    out.evidence_status = static_cast<std::uint32_t>(evidence.status);
    out.epoch = evidence.epoch; out.sequence = evidence.sequence; out.camera_sequence = evidence.camera_sequence;
    out.tick = evidence.tick; out.camera_tick = evidence.camera_tick; out.loss_revision = evidence.loss_revision;
    out.frame_kind = static_cast<std::uint32_t>(evidence.frame.kind);
    out.frame_generation = evidence.frame.generation; out.frame_numeric = evidence.frame.numeric;
    out.frame_token = evidence.frame.token; out.frame_has_numeric = evidence.frame.has_numeric;
    out.viewport = evidence.viewport;
    observation_level level = observation_level::unavailable;
    switch (evidence.status) {
      case sl::evidence_status::source_associated_evaluation: level = observation_level::source; break;
      case sl::evidence_status::command_associated_evaluation: level = observation_level::command; break;
      case sl::evidence_status::tracked_content_evaluation: level = observation_level::content; break;
      default: return;
    }
    const auto &expected = target.selection;
    const auto &source = expected.original;
    if (!detail::valid_selection(expected) || !target.camera.unit_epoch || !(target.camera.source == source) || !selected.ready ||
        selected.resource != source.native || selected.source_id != source.lifetime ||
        selected.layout_epoch != expected.layout_epoch || selected.frame_index != target.capture_frame ||
        selected.width != source.width || selected.height != source.height || selected.x != source.left ||
        selected.y != source.top || selected.active_width != source.extent_width || selected.active_height != source.extent_height ||
        !(expected.crop == rectangle{source.left, source.top, source.extent_width, source.extent_height}) ||
        selected.backup_resource != expected.sampled.native || selected.backup_id != expected.sampled.lifetime ||
        !same_selection_observation(selected, evidence.selection)) {
      out.binding = observation_binding_status::selection_mismatch; return;
    }
    if (!same_copy_key(target.copy, observed_copy_key(selected.depth_copy)) ||
        !same_copy_key(target.copy, observed_copy_key(evidence.selection.depth_copy)) ||
        (selected.depth_copy.source.resource.native &&
          (selected.depth_copy.source.resource.native != source.native || selected.depth_copy.source.resource.lifetime != source.lifetime)) ||
        (selected.depth_copy.backup.native &&
          (selected.depth_copy.backup.native != expected.sampled.native || selected.depth_copy.backup.lifetime != expected.sampled.lifetime)) ||
        !sl::content::same(selected.depth_copy.source.resource, evidence.selection.depth_copy.source.resource) ||
        !sl::content::same(selected.depth_copy.backup, evidence.selection.depth_copy.backup) ||
        !same_marker(selected.capture_marker, evidence.selection.capture_marker) ||
        !same_marker(selected.depth_copy.recording, evidence.selection.depth_copy.recording)) {
      out.binding = observation_binding_status::copy_mismatch; return;
    }
    if (!evidence.successful_evaluation || !evidence.frame_correlated || !evidence.sequence || !evidence.epoch ||
        evidence.decoded != sl::decode_status::ok || !evidence.projection.valid() ||
        !std::isfinite(evidence.projection.depth_offset) || !std::isfinite(evidence.projection.depth_scale) ||
        evidence.projection.depth_scale == 0 || evidence.matched_tag >= 2 ||
        !evidence.tags[evidence.matched_tag].present || !evidence.tags[evidence.matched_tag].supported ||
        evidence.tags[evidence.matched_tag].value.native_resource != source.native ||
        (evidence.tags[evidence.matched_tag].value.type != 0 && evidence.tags[evidence.matched_tag].value.type != 48) ||
        evidence.tags[evidence.matched_tag].value.viewport != evidence.viewport) {
      out.binding = observation_binding_status::rejected_camera; return;
    }
    if ((evidence.frame.kind != sl::frame_identity_kind::v1_numeric &&
         evidence.frame.kind != sl::frame_identity_kind::v2_observed_token) ||
        (evidence.frame.kind == sl::frame_identity_kind::v1_numeric && !evidence.frame.has_numeric) ||
        (evidence.frame.kind == sl::frame_identity_kind::v2_observed_token && !evidence.frame.generation)) {
      out.binding = observation_binding_status::rejected_frame; return;
    }
    out.tag_type = evidence.tags[evidence.matched_tag].value.type;
    out.binding = observation_binding_status::captured;
    out.level = level;
    target.camera.A = evidence.projection.depth_offset;
    target.camera.B = evidence.projection.depth_scale;
    // camera.frame stays the capture owner's frame. The independent Streamline
    // identity above must not be disguised as a proven ReShade frame mapping.
    target.projection_associated = true;
  }
}
