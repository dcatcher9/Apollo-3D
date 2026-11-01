// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "scene_depth_source.h"
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
    // DLSS-G's own evaluation (presentation_scope.h) supplies FG depth
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
    // A silent FG scope's epoch and viewport (select_frame_generation_scope):
    // it selects as without FG, and its viewport still breaks a tie of
    // active Streamline views, as a required scope's does (owner ruling R5,
    // select_latest_capture). Zero silent_epoch: none.
    std::uint64_t silent_epoch{};
    std::uint32_t silent_viewport{};
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
}

namespace sunshine_game3d::depth_output {
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
}
