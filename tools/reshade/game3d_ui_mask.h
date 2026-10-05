// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "game3d_ui_ticket.h"
#include "streamline_buffer_types.h"
#include "streamline_depth_capture.h"

// A live, bounded owner for SL's dedicated UI alpha/color, real backbuffer and
// HUD-less scene color. HUD-less pixels require comparison with paired final color.
// It shares the native snapshot/retirement machinery with depth, but never
// nominates a depth source or interprets the alpha pixels on the CPU.
namespace sunshine_game3d::ui_mask {
  // Values are canonical Streamline buffer types, so dumps record the same tag number.
  enum class source_kind : std::uint32_t {
    backbuffer = sunshine_streamline::buffers::backbuffer,
    color_and_alpha = sunshine_streamline::buffers::ui_color_and_alpha,
    alpha = sunshine_streamline::buffers::ui_alpha,
    hudless = sunshine_streamline::buffers::hudless_color
  };
  constexpr std::uint32_t source_mask(source_kind kind) {
    return kind == source_kind::backbuffer ? 1u : kind == source_kind::color_and_alpha ? 2u :
      kind == source_kind::alpha ? 4u : kind == source_kind::hudless ? 8u : 0u;
  }
  inline constexpr std::uint32_t all_sources = source_mask(source_kind::backbuffer) |
    source_mask(source_kind::color_and_alpha) | source_mask(source_kind::alpha) | source_mask(source_kind::hudless);
  inline bool supported_format(source_kind kind, std::uint32_t format) {
    if (kind == source_kind::alpha) return format == 41 || format == 54 || format == 56 || format == 61;
    return (kind == source_kind::backbuffer || kind == source_kind::color_and_alpha || kind == source_kind::hudless) &&
      (format == 2 || format == 10 || format == 24 || format == 28 || format == 29 || format == 87 || format == 91);
  }
  struct request {
    std::uint64_t runtime{}, device_identity{}, epoch{}, revision{};
    std::uint32_t viewport{}, width{}, height{};
    bool enabled{};
    // Active candidates constrain capture before source ranking.
    // Changing this set revokes all completed and in-flight work in the scope.
    std::uint32_t allowed_kinds = all_sources;
  };
  struct boundary {
    sunshine_scene_depth::frame source;
    std::uint64_t command{};
    std::uint32_t tag_scope{}; // 0 global, 1 explicit frame; never inferred from Present.
    source_kind kind = source_kind::backbuffer;
    std::uint64_t source_present_generation{}; // Captured at this exact tag entry.
  };
  // The presented color a HUD-less image is compared with when no same-batch
  // Backbuffer was tagged with it: the Present after its tag (next_present,
  // presents_ago 0), or, for a capture that completed later (another queue),
  // the color retained up to max_late_presents Presents ago (earlier_present);
  // anything older is unpaired. Counting Presents only proposes this pair: a
  // game may tag its next frame before the Present counted for it, and frame
  // generation presents interpolated color between real frames, so the pair is
  // never exact (UI framework E2). Only V2's test of the pixels validates it,
  // and only a same-batch Backbuffer (same_tag_interval) makes a pair exact.
  enum class hudless_present { unpaired, next_present, earlier_present };
  inline constexpr std::uint32_t max_late_presents = 2;
  struct hudless_pairing {
    hudless_present kind = hudless_present::unpaired;
    std::uint32_t presents_ago = 0; // Nonzero only for earlier_present.
  };
  constexpr hudless_pairing pair_hudless_present(std::uint64_t tagged, std::uint64_t current) {
    if (!tagged || tagged == UINT64_MAX || current <= tagged) return {};
    const auto ago = current - tagged - 1;
    if (!ago) return {hudless_present::next_present};
    if (ago > max_late_presents) return {};
    return {hudless_present::earlier_present, static_cast<std::uint32_t>(ago)};
  }
  // Present counting of generated Presents (UI framework T1). Under frame
  // generation a game may tag its UI inputs (UIAlpha, the UI color tag, the
  // Backbuffer) for real frames only, and a generated Present may then offer
  // none of them: Expedition 33's Backbuffer at 2x, 3x or 4x. Such a Present
  // within the reported generated count (one while the count is unknown) of
  // the last Present that offered one shows that Present's real frame, so it
  // is generated and shows that decision. Presents are numbered by the
  // caller, one per render; zero is none. A Present that offers any input (a
  // HUD-less image or the offscreen UI layer included), or one without frame
  // generation or beyond the count, is real, and when it misses an accepted
  // input the T1 grace reuses the previous real decision once. No multiplier
  // constant: a count reported too high holds only Presents without input of
  // their own, and one too low leaves the later ones to the grace.
  constexpr bool generated_without_input(std::uint64_t present, std::uint64_t input_present, bool fg_active,
      std::uint32_t generated_frames) {
    if (!fg_active || !input_present || present <= input_present) return false;
    const std::uint64_t generated = !generated_frames ? 1u : generated_frames;
    return present - input_present <= generated;
  }
  static_assert(generated_without_input(11, 10, true, 3) && generated_without_input(13, 10, true, 3) &&
    !generated_without_input(14, 10, true, 3) && generated_without_input(11, 10, true, 0) &&
    !generated_without_input(12, 10, true, 0) && !generated_without_input(11, 10, false, 3) &&
    !generated_without_input(11, 0, true, 3) && !generated_without_input(10, 10, true, 3));
  // Two captures belong to one tag batch (one game frame) when both were made
  // in the same Present interval. Each source keeps its own Present counter, so
  // compare Presents since each tag rather than the raw counter values.
  constexpr bool same_tag_interval(std::uint64_t tagged_a, std::uint64_t current_a,
      std::uint64_t tagged_b, std::uint64_t current_b) {
    const auto valid = [](std::uint64_t tagged, std::uint64_t current) {
      return tagged && tagged != UINT64_MAX && current >= tagged;
    };
    return valid(tagged_a, current_a) && valid(tagged_b, current_b) && current_a - tagged_a == current_b - tagged_b;
  }
  struct selection {
    sunshine_streamline::depth_capture::diagnostic_ticket ticket;
    sunshine_streamline::depth_capture::diagnostic_texture texture;
    boundary origin;
    // Read from the retained original resource when immutable pixels are polled.
    // A generation match is a necessary pairing gate, not a content/frame proof.
    std::uint64_t current_source_present_generation{};
    // S3: the probe's token generation of the tag's frame (Streamline contract
    // 6: unique per slGetNewFrameToken issue, numbered or not; 0 none).
    std::uint64_t token_generation{};
  };
  enum class capture_gate { not_observed, invalid_viewport, inactive_observation, observation_changed,
    no_matching_request, ambiguous_request, no_ui_tags, admitted };
  inline const char *name(capture_gate value) {
    switch (value) {
      case capture_gate::invalid_viewport: return "invalid_viewport";
      case capture_gate::inactive_observation: return "inactive_observation";
      case capture_gate::observation_changed: return "observation_changed";
      case capture_gate::no_matching_request: return "no_matching_request";
      case capture_gate::ambiguous_request: return "ambiguous_request";
      case capture_gate::no_ui_tags: return "no_ui_tags";
      case capture_gate::admitted: return "admitted";
      default: return "not_observed";
    }
  }
  struct capture_gate_observation {
    capture_gate state{};
    std::uint64_t epoch{}, revision{}, sequence{}, tick{};
    std::uint32_t viewport{}, seen_kinds{}, matching_requests{};
  };
  struct diagnostic_snapshot {
    request wanted;
    // Latest global hook gate, independent of this request's matching boundary.
    // Keeping it across request replacement exposes mismatched or reset scopes.
    capture_gate_observation hook_gate;
    std::uint64_t request_generation{};
    boundary latest_boundary;
    sunshine_streamline::depth_capture::record_diagnostic record;
    bool record_attempted{}, record_completed{}, sdk_result_known{}, sdk_successful{};
    // S3: where begin() stopped without an attempt, the latest in this request
    // (none after an attempt), and the process's counts per stage since start
    // (ui_ticket::begin_stage; no_request and ambiguous_request have no
    // request of their own, so only the counts name them).
    sunshine_game3d::ui_ticket::begin_stage begin_refusal{};
    std::array<std::uint64_t, std::size_t(sunshine_game3d::ui_ticket::begin_stage::count)> begin_refusals{};
  };

  // Repeating an identical request preserves completed pixels. Disable or any
  // scope/size change revokes them, including in-flight attempts from the old scope.
  void set_request(const request &value);
  void invalidate(std::uint64_t runtime);
  void invalidate_all();
  void invalidate_scope(std::uint64_t epoch, std::uint32_t viewport);
  // Strict diagnostic query exposes completed immutable pixels within
  // maximum_source_age_ms. The consumer must still use copy_diagnostic_texture.
  bool acquire(std::uint64_t runtime, selection &out, std::uint64_t now_ms);
  // Candidates retain independent completed/pending snapshots. A resource's
  // availability or priority cannot hide another kind from automatic validation.
  // A nonzero consumer_queue opts into ordered local GPU acquisition; it must
  // then use copy_local_texture on that queue. Zero keeps strict diagnostics.
  bool acquire_kind(std::uint64_t runtime, source_kind kind, selection &out, std::uint64_t now_ms,
    std::uint64_t consumer_queue = 0);
  // S3 token pairing (shadow until ui_ticket::identity_authoritative): of the
  // retained HUD-less and Backbuffer snapshots (two per kind), the pair of one
  // game frame (ui_ticket::same_frame: same epoch and viewport and the same
  // nonzero token generation) with the newest token generation whose both
  // snapshots are ready for this consumer queue (as acquire_kind). Read-only:
  // it retires nothing, so a caller asks for it before acquire_kind retires
  // an older ready snapshot. False when no such pair is ready.
  bool acquire_batch(std::uint64_t runtime, std::uint64_t now_ms, std::uint64_t consumer_queue, selection &hudless,
    selection &backbuffer);
  // Plain metadata for the newest boundary in the current requested scope.
  // True with a zero boundary sequence means no matching tag has arrived yet;
  // record_attempted distinguishes a native call from no call; record_completed
  // is true only when that call returned its diagnostic. A pending native call
  // must not be classified as a rejection using its default record fields.
  // This does not poll the GPU, retain a source lease, or authorize any pixels.
  bool query_diagnostic(std::uint64_t runtime, diagnostic_snapshot &out);

  // Streamline adapter boundary. No source reference is obtained when no live
  // request matches. Every begin attempt must finish after the original SDK call,
  // even if native recording had no free slot. Null/invalid tags revoke that kind;
  // dedicated alpha, UI color and backbuffer remain distinct semantic sources.
  bool interested(std::uint64_t epoch, std::uint64_t revision, std::uint32_t viewport,
    std::uint32_t *matching_requests = nullptr);
  void observe_gate(const capture_gate_observation &value);
  struct attempt {
    std::uint64_t runtime{}, generation{}, sequence{}, reservation{};
    sunshine_streamline::depth_capture::diagnostic_ticket ticket;
    source_kind kind = source_kind::backbuffer;
  };
  attempt begin(const boundary &where, const sunshine_streamline::depth_capture::input &input);
  void finish(const attempt &value, bool successful);
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
  namespace testing {
    bool last_attempt(std::uint64_t runtime, boundary &out);
    // Whether the last attempt's record asked for the S3 token clock write.
    bool last_token_clock(std::uint64_t runtime);
  }
#endif
}
