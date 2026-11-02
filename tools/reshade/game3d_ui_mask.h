// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "freshness.h"
#include "streamline_buffer_types.h"
#include "streamline_camera_data.h"
#include "streamline_depth_capture.h"
#include "game3d_ui_selection.h"
#include "ui_evidence.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

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
  // The capture slot order of every per-kind array (the renderer's candidate
  // slots 0-3): UIAlpha, the UI color tag, Backbuffer, HUD-less.
  inline constexpr std::array<source_kind, 4> capture_order{source_kind::alpha, source_kind::color_and_alpha,
    source_kind::backbuffer, source_kind::hudless};
  constexpr unsigned capture_slot(source_kind kind) {
    return kind == source_kind::alpha ? 0u : kind == source_kind::color_and_alpha ? 1u :
      kind == source_kind::backbuffer ? 2u : 3u;
  }
  // Which origin supplies each capture slot of one frame-generation scope
  // (epoch, viewport): the game's Streamline UI tags or DLSS-G's own inputs of
  // the same kind (streamline_camera_probe.cpp, which guards it). copied: the
  // tick the scope's newest tag copy of the kind was recorded, zero once one
  // was refused or its SDK call failed. refused: the tick the kind's newest
  // copy deferred to its frame's Present (R2, the matrix's U2 and U5) was
  // refused, zero once one is made. A recorded copy or a deferred one of
  // another scope starts that scope's entry. Ticks are game-clock ms.
  struct tag_precedence {
    std::uint64_t epoch{};
    std::uint32_t viewport{};
    std::array<std::uint64_t, 4> copied{}, refused{};

    // A tag copy of slot recorded or refused at now; deferred: one made at
    // its frame's Present.
    void remember(std::uint64_t scope_epoch, std::uint32_t scope_viewport, unsigned slot, bool recorded, bool deferred,
        std::uint64_t now) {
      if (slot >= copied.size()) return;
      if ((recorded || deferred) && !scope(scope_epoch, scope_viewport)) *this = {scope_epoch, scope_viewport, {}, {}};
      if (!scope(scope_epoch, scope_viewport)) return;
      copied[slot] = recorded ? now : 0;
      if (deferred) refused[slot] = recorded ? 0 : now;
    }
    // A kept copy's SDK call failed: its kind no longer has precedence.
    void forget(unsigned slot) {
      if (slot < copied.size()) copied[slot] = 0;
    }
    // The kind's newest deferred copy was refused within the source age: its
    // tags defer no copy of it (one would only be refused again while DLSS-G's
    // own copies hold its snapshots) and DLSS-G's input supplies it, until
    // the refusal ages and a tag defers again.
    bool deferral_refused(std::uint64_t scope_epoch, std::uint32_t scope_viewport, unsigned slot, std::uint64_t now) const {
      return slot < refused.size() && scope(scope_epoch, scope_viewport) && refused[slot] &&
        sunshine_freshness::ahead_tolerant_lt(refused[slot], now, sunshine_freshness::source_age_ms);
    }
    // Whether DLSS-G's input of the kind stands aside for the tags': their
    // copy was recorded within the source age, or one waits for this frame's
    // Present (pending) and the kind's deferrals are not refused.
    bool tags_supply(std::uint64_t scope_epoch, std::uint32_t scope_viewport, unsigned slot, bool pending,
        std::uint64_t now) const {
      if (slot >= copied.size()) return false;
      if (scope(scope_epoch, scope_viewport) && copied[slot] &&
          sunshine_freshness::ahead_tolerant_lt(copied[slot], now, sunshine_freshness::source_age_ms))
        return true;
      return pending && !deferral_refused(scope_epoch, scope_viewport, slot, now);
    }

  private:
    bool scope(std::uint64_t scope_epoch, std::uint32_t scope_viewport) const {
      return epoch == scope_epoch && viewport == scope_viewport;
    }
  };
  // UIAlpha is a single-channel image of the UI's opacity (the DLSS-G guide):
  // R32_FLOAT, R16_FLOAT, R16_UNORM and R8_UNORM carry it in red, A8_UNORM in
  // alpha (ui_alpha_in_alpha; the detection reads it there through
  // ui_detection::stored_ui_alpha_in_alpha). A typeless single-channel
  // allocation is one of them only through its declared nativeFormat
  // (depth_capture::auxiliary_snapshot_format).
  constexpr bool ui_alpha_format(std::uint32_t format) {
    return format == 41 || format == 54 || format == 56 || format == 61 || format == 65;
  }
  constexpr bool ui_alpha_in_alpha(std::uint32_t format) { return format == 65; }
  // A snapshot's typed DXGI format (depth_capture::auxiliary_snapshot_format of
  // the tagged resource's format and declared nativeFormat) that the kind can
  // be read from: UIAlpha one of its single-channel formats; HUD-less any
  // colour format ui_selection::detail::classify knows; Backbuffer and the UI
  // color tag only those with alpha. Unknown formats stay refused.
  constexpr bool supported_format(source_kind kind, std::uint32_t format) {
    if (kind == source_kind::alpha) return ui_alpha_format(format);
    const auto colour = ui_selection::detail::classify(format);
    if (colour.kind == ui_selection::detail::family::unknown) return false;
    return kind == source_kind::hudless ||
      ((kind == source_kind::backbuffer || kind == source_kind::color_and_alpha) && colour.has_alpha);
  }
  // The gate and the classifier agree on every DXGI format, and every typeless
  // allocation is judged by its resolved snapshot format: HUD-less admits
  // 2, 10, 11, 24, 26, 28, 29, 87, 88, 91 and 93; Backbuffer and UIColorAndAlpha
  // those of them with alpha; UIAlpha 41, 54, 56, 61 and 65.
  static_assert([] {
    namespace selection = ui_selection::detail;
    namespace capture = sunshine_streamline::depth_capture;
    std::uint32_t hudless = 0, alpha = 0, ui_alpha = 0;
    for (std::uint32_t f = 0; f <= 132; ++f) {
      const auto c = selection::classify(f);
      const bool known = c.kind != selection::family::unknown;
      if (supported_format(source_kind::hudless, f) != known ||
          supported_format(source_kind::backbuffer, f) != (known && c.has_alpha) ||
          supported_format(source_kind::color_and_alpha, f) != (known && c.has_alpha) ||
          supported_format(source_kind::alpha, f) != ui_alpha_format(f) ||
          (capture::typeless_color(f) && (known || ui_alpha_format(f))))
        return false;
      // A resolved snapshot format is never typeless.
      if (capture::typeless_color(capture::auxiliary_snapshot_format(f)) ||
          capture::typeless_color(capture::auxiliary_snapshot_format(f, f + 1)))
        return false;
      hudless += supported_format(source_kind::hudless, f) ? f : 0;
      alpha += supported_format(source_kind::backbuffer, f) ? f : 0;
      ui_alpha += supported_format(source_kind::alpha, f) ? f : 0;
    }
    return hudless == 2 + 10 + 11 + 24 + 26 + 28 + 29 + 87 + 88 + 91 + 93 &&
      alpha == 2 + 10 + 11 + 24 + 28 + 29 + 87 + 91 && ui_alpha == 41 + 54 + 56 + 61 + 65;
  }());
  // A request is live while its runtime refreshes it (set_request on every
  // Present that wants UI captures): only requests refreshed within
  // the source age (freshness.h) of a tag match it, so a swapchain that stopped
  // presenting (a launcher, an old swapchain beside its replacement) never
  // makes the tags of the live one ambiguous.
  struct request {
    std::uint64_t runtime{}, device_identity{}, epoch{}, revision{};
    std::uint32_t viewport{}, width{}, height{};
    bool enabled{};
    // Active candidates constrain capture before source ranking.
    // Changing this set revokes all completed and in-flight work in the scope.
    std::uint32_t allowed_kinds = all_sources;
  };
  // A tag is captured only while its runtime reads UI captures (acquire_all)
  // within this window; otherwise begin() refuses it (no_reader) instead of
  // copying pixels nobody reads (a Present that requested captures but
  // returned before its render).
  inline constexpr std::uint64_t reader_window_ms = 1000;
  struct boundary {
    sunshine_scene_depth::frame source;
    std::uint64_t command{};
    // 0 global, 1 explicit frame, 2 DLSS-G's own NGX evaluation, 3 a local
    // input of a Streamline evaluate; never inferred from Present.
    std::uint32_t tag_scope{};
    source_kind kind = source_kind::backbuffer;
    std::uint64_t source_present_generation{}; // Captured at this exact tag entry.
    // The tagged resource's DXGI format (depth_capture::retain_source; 0
    // unknown). begin() resolves it with the tag's declared nativeFormat
    // (input.resource.native_format, depth_capture::auxiliary_snapshot_format)
    // and refuses a snapshot format the kind cannot be read from (format)
    // before any copy.
    std::uint32_t format{};
    // An unsupported tag's cause, which names begin()'s refusal stage.
    sunshine_streamline::tag_refusal refusal{};
    // Read from DLSS-G's own NGX evaluation (feature 11: DLSSG.Backbuffer,
    // HUDLess, UI, UIAlpha), not from a Streamline tag. Its sequence is that
    // evaluation's, shared by all of its inputs. Each origin revokes only its
    // own snapshots: Assassin's Creed Shadows (10-23) tags its HUD-less and
    // Backbuffer without a command list on every frame (copied before the
    // frame's Present since Stage 3 fix 2; before, refused as shape), which
    // must not erase the copies DLSS-G's evaluation recorded.
    bool frame_generation_evaluation{};
  };
  // Pure pairing/Present evidence is owned by ui_evidence.h. Preserve these
  // public access paths for capture consumers.
  using ui_evidence::hudless_present;
  using ui_evidence::max_late_presents;
  using ui_evidence::hudless_pairing;
  using ui_evidence::pair_hudless_present;
  using ui_evidence::pair_hudless_offer;
  // The count fallback of ui_evidence::generated_in_window only.
  using ui_evidence::generated_without_input;
  using ui_evidence::snapshot_input_slots;
  using ui_evidence::snapshot_inputs;
  using ui_evidence::same_snapshot_inputs;

  // Whether a ready snapshot of this kind may be offered to detection
  // (ui_input::acquire). An unauthenticated UIAlpha tag cannot hide
  // independently usable lower-priority candidates in Auto; an authenticated
  // one is validated on the GPU, as is DLSS-G's named DLSSG.UIAlpha
  // (frame_generation_evaluation). A manual choice may use it. A HUD-less
  // image needs a color to pair with (paired).
  enum class candidate_refusal { none, ui_alpha_unauthenticated, hudless_unpaired };
  constexpr candidate_refusal admit_candidate(source_kind kind, bool manual, bool alpha_authenticated,
      bool frame_generation_evaluation, bool paired) {
    if (kind == source_kind::hudless && !paired) return candidate_refusal::hudless_unpaired;
    return manual || kind != source_kind::alpha || alpha_authenticated || frame_generation_evaluation ?
      candidate_refusal::none : candidate_refusal::ui_alpha_unauthenticated;
  }
  using ui_evidence::same_tag_interval;
  // An exact pair (UI framework E2): two snapshots of one tag batch, Streamline
  // tags of one Present interval (same_tag_interval), or DLSS-G inputs of one
  // NGX evaluation (its single sequence and epoch, captured at one call). An
  // input of DLSS-G's evaluation never pairs exactly with a Streamline tag.
  inline bool same_batch(const boundary &a, std::uint64_t current_a, const boundary &b, std::uint64_t current_b) {
    if (a.frame_generation_evaluation != b.frame_generation_evaluation) return false;
    if (a.frame_generation_evaluation &&
        (!a.source.sequence || a.source.sequence != b.source.sequence || a.source.epoch != b.source.epoch))
      return false;
    return same_tag_interval(a.source_present_generation, current_a, b.source_present_generation, current_b);
  }
  struct selection {
    sunshine_streamline::depth_capture::diagnostic_ticket ticket;
    sunshine_streamline::depth_capture::diagnostic_texture texture;
    boundary origin;
    // Read from the retained original resource when immutable pixels are polled.
    // A generation match is a necessary pairing gate, not a content/frame proof.
    std::uint64_t current_source_present_generation{};
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
  // Where a tag's begin() stopped without a capture attempt: no live request
  // matches, more than one does, the request's kinds exclude the tag, the
  // tag is not newer than the request's latest, its shape or declared
  // lifetime is unsupported, both of its kind's reservations are taken, its
  // runtime has not read UI captures within reader_window_ms, its format
  // cannot be read as its kind (supported_format), or the tag itself was
  // unsupported: a structure GUID or version or an unreadable tag
  // (header_version), an unknown chained structure or PrecisionInfo
  // (extension), or a resource type other than eTex2d/eUnknown (eUnknown only
  // where the hooked release declares it, S12).
  // The capture gate line counts them per stage.
  enum class begin_stage : std::uint8_t {
    none,
    no_request,
    ambiguous_request,
    kind_filtered,
    not_newer,
    shape,
    unsupported_lifetime,
    no_reservation,
    no_reader,
    format,
    header_version,
    extension,
    resource_type,
    count
  };
  inline constexpr std::array<std::string_view, std::size_t(begin_stage::count)> begin_stage_names{"none", "no_request",
    "ambiguous_request", "kind_filtered", "not_newer", "shape", "unsupported_lifetime", "no_reservation", "no_reader",
    "format", "header_version", "extension", "resource_type"};
  constexpr std::string_view name(begin_stage value) {
    return value < begin_stage::count ? begin_stage_names[std::size_t(value)] : std::string_view("none");
  }
  struct diagnostic_snapshot {
    request wanted;
    // Latest global hook gate, independent of this request's matching boundary.
    // Keeping it across request replacement exposes mismatched or reset scopes.
    capture_gate_observation hook_gate;
    std::uint64_t request_generation{};
    boundary latest_boundary;
    sunshine_streamline::depth_capture::record_diagnostic record;
    bool record_attempted{}, record_completed{}, sdk_result_known{}, sdk_successful{};
    // Where begin() stopped without an attempt, the latest in this request
    // (none after an attempt), and the process's counts per stage since start
    // (no_request and ambiguous_request have no request of their own, so
    // only the counts name them).
    begin_stage begin_refusal{};
    std::array<std::uint64_t, std::size_t(begin_stage::count)> begin_refusals{};
  };

  // Repeating an identical request preserves completed pixels and refreshes
  // it (now_ms). Disable or any scope/size change revokes them, including
  // in-flight attempts from the old scope. A runtime without an entry takes a
  // free one, else the one refreshed longest ago once it expired.
  void set_request(const request &value, std::uint64_t now_ms);
  void invalidate(std::uint64_t runtime);
  void invalidate_all();
  void invalidate_scope(std::uint64_t epoch, std::uint32_t viewport);
  // The newest ready snapshot of each kind in capture_order (by[i] is valid
  // when found has source_mask(capture_order[i])).
  struct acquired {
    std::array<selection, 4> by{};
    std::uint32_t found{};
  };
  // One scan per Present: every snapshot within the source age is
  // polled once, older and failed ones retire, and each allowed kind's newest
  // ready snapshot is returned; a kind's availability or priority cannot hide
  // another kind from automatic validation. A nonzero consumer_queue opts
  // into ordered local GPU acquisition; it must then use copy_local_texture
  // (or lease_local_view) on that queue. Zero keeps strict diagnostics. Each
  // call marks the runtime a reader (reader_window_ms).
  bool acquire_all(std::uint64_t runtime, std::uint64_t now_ms, acquired &out, std::uint32_t allowed = all_sources,
    std::uint64_t consumer_queue = 0);
  // Plain metadata for the newest boundary in the current requested scope.
  // True with a zero boundary sequence means no matching tag has arrived yet;
  // record_attempted distinguishes a native call from no call; record_completed
  // is true only when that call returned its diagnostic. A pending native call
  // must not be classified as a rejection using its default record fields.
  // This does not poll the GPU, retain a source lease, or authorize any pixels.
  bool query_diagnostic(std::uint64_t runtime, diagnostic_snapshot &out);

  // Streamline adapter boundary. No source reference is obtained when no live
  // request matches. Every begin attempt must finish after the original SDK call,
  // even if native recording had no free slot. Null/invalid tags (a resource of
  // another shape or an unreadable format included) revoke that kind's
  // snapshots of the same origin (a Streamline tag or DLSS-G's evaluation);
  // dedicated alpha, UI color and backbuffer remain distinct semantic sources.
  // now_ms is the tag's entry tick; only requests refreshed within the
  // source age of it count.
  bool interested(std::uint64_t epoch, std::uint64_t revision, std::uint32_t viewport, std::uint64_t now_ms,
    std::uint32_t *matching_requests = nullptr);
  void observe_gate(const capture_gate_observation &value);
  struct attempt {
    std::uint64_t runtime{}, generation{}, sequence{}, reservation{};
    sunshine_streamline::depth_capture::diagnostic_ticket ticket;
    source_kind kind = source_kind::backbuffer;
    // Where begin() stopped without a copy attempt (none when it attempted
    // one), and, for an attempt, the native record's stage (recorded when it
    // returned a ticket). Diagnostics only.
    begin_stage refusal{};
    sunshine_streamline::depth_capture::record_stage record{};
    // The snapshot's origin (boundary::frame_generation_evaluation): a failed
    // SDK call revokes only its own origin's inputs.
    bool frame_generation_evaluation{};
  };
  attempt begin(const boundary &where, const sunshine_streamline::depth_capture::input &input);
  void finish(const attempt &value, bool successful);
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
  namespace testing {
    bool last_attempt(std::uint64_t runtime, boundary &out);
  }
#endif
}
