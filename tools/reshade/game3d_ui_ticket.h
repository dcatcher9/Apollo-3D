// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// The S3 snapshot ticket (docs/reshade-sbs.md, UI decision framework: "S3
// snapshot ticket (shadow)", rules E1, E2 and T1). One immutable ticket rides
// beside every UI candidate snapshot (a Streamline UIAlpha, UIColorAndAlpha,
// Backbuffer or HUD-less tag copy, the offscreen UI layer's before-clear copy,
// live or census, the add-on's current-colour and retained-Present copies, a
// Dump 3D capture) and records its frame labels and their proof.
//
// Labels come from two add-on clocks per device (game3d_frame_clock.h), each a
// monotonic 32-bit GPU word where 0 means never written:
// - the present clock C_P: frame_clock::advance writes the next Present label
//   on the presenting queue at every foreground Present, before the Game 3D
//   render; everything the add-on copies at that Present carries that label;
// - the token clock C_T: the Streamline capture owner writes the low 32 bits
//   of the probe's token generation in the game's own list, right after each
//   Backbuffer tag snapshot.
// A snapshot recorded into a game list is followed, in the same list, by two
// 4-byte copies of C_P and C_T into its 16-byte stamp entry (stamp_bytes), so
// the image and its stamp are written by the same execution: a list that runs
// late, twice or never leaves them consistent.
//
// A label is valid in one space (token or present). Present labels by
// boundary (present_label): a before_clear copy holds the frame of the Present
// whose label it read; an at_tag copy belongs to the frame after the Present
// it read (read + 1); a present copy is the Present's own label. A read of 0
// is unstamped. A game copy's present label is valid only on the presenting
// queue and over a real span (real_span); its token label only on the
// Backbuffer producer's queue. There is one pairing rule (pair_exact): an
// image and its final-image reference pair exactly iff they carry equal valid
// labels in one space; otherwise the pair is absent with a refusal, never
// inexact. Labels are 32-bit and the token label keeps the low 32 bits of the
// token generation: wraparound or collision is a known, negligible limit.
//
// S3 ships in shadow: tickets are logged, counted, dumped and reported beside
// today's Present-counting pairing (identity_shadow); nothing decides from a
// ticket while identity_authoritative is false. This header owns the log text
// after "Sunshine UI identity: runtime=<p> " (format_identity_counters) and
// after "Sunshine FG interposers: runtime=<p> " (format_interposers). No
// ReShade dependency.
#include "game3d_ui_selection.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace sunshine_game3d::ui_ticket {
  // The S3 enable switch for HUD-less (source 5) exactness by token and T1
  // identity by token, flipped once after the live shadow evidence
  // docs/reshade-sbs.md lists (UI decision framework, S3 snapshot ticket).
  // The pre-UI change set (source 12) rides on UIPinOnlyUI instead.
  inline constexpr bool identity_authoritative = false;

  // Whether identity decides: the constant, or the sequence harness's
  // test-only override (ui_temporal::detection_state::identity_override).
  constexpr bool authoritative(bool test_override) {
    return identity_authoritative || test_override;
  }

  // Ticket and stamp slots of ui_detection_inputs::tickets and the renderer's
  // stamp buffer: 0-2 the UIAlpha, UI colour and Backbuffer tags, 3 the
  // presented colour, 4 the offscreen UI layer, 5 the HUD-less tag.
  namespace slot {
    inline constexpr std::size_t ui_alpha = 0, ui_color = 1, backbuffer = 2, current = 3, layer = 4, hudless = 5, count = 6;
  }  // namespace slot

  // The ticket slot of a renderer candidate slot (0-2 tags, 3 HUD-less, 4 the
  // layer); slot::count for none.
  constexpr std::size_t slot_of_candidate(unsigned candidate_slot) {
    if (candidate_slot < 3) {
      return candidate_slot;
    }
    if (candidate_slot == 3) {
      return slot::hudless;
    }
    return candidate_slot == 4 ? slot::layer : slot::count;
  }

  // One stamp entry: the C_P read at byte 0, the C_T read at byte 4.
  inline constexpr std::uint64_t stamp_bytes = 16, stamp_present_offset = 0, stamp_token_offset = 4;

  // What a ticket was captured from.
  enum class capture_kind : std::uint8_t {
    sl_tag,  // A Streamline tag's snapshot.
    layer_copy,  // The offscreen UI layer's before-clear copy (live or census).
    present_color,  // The add-on's copy of presented colour (current or retained).
    dump,  // A Dump 3D capture.
    count
  };
  inline constexpr std::array<std::string_view, std::size_t(capture_kind::count)> capture_kind_names {"sl_tag", "layer_copy", "present_color", "dump"};

  // Which frame a copy holds relative to the Present label it read.
  enum class boundary : std::uint8_t {
    at_tag,  // At the tag: the frame after the Present read (read + 1).
    before_clear,  // Before the layer's clear: the frame of the Present read.
    present,  // The presented image: its own Present label.
    count
  };
  inline constexpr std::array<std::string_view, std::size_t(boundary::count)> boundary_names {"at_tag", "before_clear", "present"};

  enum class label_space : std::uint8_t {
    none,
    token,
    present,
    count
  };
  inline constexpr std::array<std::string_view, std::size_t(label_space::count)> label_space_names {"none", "token", "present"};

  // Why a ticket, a label or a pair proves nothing. The order is the log's;
  // first_refusal gives the precedence.
  enum class refusal : std::uint8_t {
    none,
    unstamped,  // The stamp read 0, or the stamp entry was recreated in a new scope.
    foreign_queue,  // Executed on a queue other than the one the label space needs: a read across queues races.
    not_real_span,  // A present label over a span with a Present not known real (FG on, unknown or suspended).
    stale_scope,  // The snapshot belongs to another scope (device, resize, epoch).
    no_reference,  // No final-image reference was proposed.
    mismatch,  // Labels differ.
    enhanced_barrier,  // The captured resource's state rests on enhanced barriers.
    render_pass,  // The layer's clear ran inside a render pass.
    recording_not_covered,  // The capture owner does not cover the recording.
    budget,  // No slot.
    begin_refused,  // A tag's begin() produced no attempt (begin_stage says why).
    count
  };
  inline constexpr std::array<std::string_view, std::size_t(refusal::count)> refusal_names {"none", "unstamped", "foreign_queue", "not_real_span", "stale_scope", "no_reference", "mismatch", "enhanced_barrier", "render_pass", "recording_not_covered", "budget", "begin_refused"};

  // Where a Streamline tag's begin() stopped without an attempt.
  enum class begin_stage : std::uint8_t {
    none,
    no_request,
    ambiguous_request,
    kind_filtered,
    not_newer,
    shape,
    unsupported_lifetime,
    no_reservation,
    count
  };
  inline constexpr std::array<std::string_view, std::size_t(begin_stage::count)> begin_stage_names {"none", "no_request", "ambiguous_request", "kind_filtered", "not_newer", "shape", "unsupported_lifetime", "no_reservation"};

  // What proves the captured resource's state at the copy: the Streamline
  // capture owner's state basis, the layer's clear pre-call, or the Present.
  enum class proof : std::uint8_t {
    none,
    sl_observed_legacy,
    sl_observed_enhanced,
    sl_declared,
    sl_contract,
    clear_precall,
    present_boundary,
    count
  };
  inline constexpr std::array<std::string_view, std::size_t(proof::count)> proof_names {"none", "sl_observed_legacy", "sl_observed_enhanced", "sl_declared", "sl_contract", "clear_precall", "present_boundary"};

  // The capture owner's state_basis (streamline_depth_capture.h) by value: 0
  // unknown, 1 observed through legacy barriers, 2 through enhanced barriers,
  // 3 declared by the tag, 4 the SDK contract.
  constexpr proof state_basis_proof(std::uint8_t basis) {
    switch (basis) {
      case 1:
        return proof::sl_observed_legacy;
      case 2:
        return proof::sl_observed_enhanced;
      case 3:
        return proof::sl_declared;
      case 4:
        return proof::sl_contract;
      default:
        return proof::none;
    }
  }

  // The queue a copy executed on relative to the queue a label space needs
  // (the presenting queue for present labels, the Backbuffer producer's for
  // token labels).
  enum class queue_relation : std::uint8_t {
    unknown,
    same,
    foreign,
    count
  };
  inline constexpr std::array<std::string_view, std::size_t(queue_relation::count)> queue_relation_names {"unknown", "same", "foreign"};

  // The GPU's identity verdict of one proposed pair (the change-set pass's
  // verdict texel, filled by the renderer's GPU verification).
  enum class gpu_verdict : std::uint8_t {
    none,
    exact,
    mismatch,
    unstamped,
    unproposed,
    count
  };
  inline constexpr std::array<std::string_view, std::size_t(gpu_verdict::count)> gpu_verdict_names {"none", "exact", "mismatch", "unstamped", "unproposed"};

  template<class Enum, std::size_t N>
  constexpr std::string_view enum_name(const std::array<std::string_view, N> &names, Enum value) {
    return std::size_t(value) < N ? names[std::size_t(value)] : std::string_view("unknown");
  }

  constexpr std::string_view name(capture_kind v) {
    return enum_name(capture_kind_names, v);
  }

  constexpr std::string_view name(boundary v) {
    return enum_name(boundary_names, v);
  }

  constexpr std::string_view name(label_space v) {
    return enum_name(label_space_names, v);
  }

  constexpr std::string_view name(refusal v) {
    return enum_name(refusal_names, v);
  }

  constexpr std::string_view name(begin_stage v) {
    return enum_name(begin_stage_names, v);
  }

  constexpr std::string_view name(proof v) {
    return enum_name(proof_names, v);
  }

  constexpr std::string_view name(queue_relation v) {
    return enum_name(queue_relation_names, v);
  }

  constexpr std::string_view name(gpu_verdict v) {
    return enum_name(gpu_verdict_names, v);
  }

  // ---------------------------------------------------------------- labels

  struct label {
    label_space space {label_space::none};
    std::uint32_t value {};
    bool valid {};
    refusal reason {refusal::unstamped};  // none exactly when valid.

    friend constexpr bool operator==(const label &a, const label &b) {
      return a.space == b.space && a.value == b.value && a.valid == b.valid && a.reason == b.reason;
    }
  };

  constexpr label valid_label(label_space space, std::uint32_t value) {
    return {space, value, true, refusal::none};
  }

  // The label without its validity, for this reason (a refused label keeps
  // its space and read value for diagnostics).
  constexpr label refuse(label l, refusal reason) {
    l.valid = false;
    l.reason = reason;
    return l;
  }

  // The present label of a copy at this boundary from its C_P read, or, for
  // the presented image, its own Present label. 0 is unstamped.
  constexpr label present_label(boundary at, std::uint32_t read, std::uint32_t own = 0) {
    if (at == boundary::present) {
      return own ? valid_label(label_space::present, own) : refuse({label_space::present, 0}, refusal::unstamped);
    }
    const std::uint32_t value = at == boundary::at_tag ? read + 1u : read;
    return read && value ? valid_label(label_space::present, value) : refuse({label_space::present, read}, refusal::unstamped);
  }

  // The token label of a game copy from its C_T read.
  constexpr label token_label_of_copy(std::uint32_t read) {
    return read ? valid_label(label_space::token, read) : refuse({label_space::token, 0}, refusal::unstamped);
  }

  // The token label of a Streamline snapshot: the low 32 bits of the probe's
  // token generation (Streamline contract 6), the value C_T carries.
  constexpr label token_label_of_tag(std::uint64_t generation) {
    return token_label_of_copy(std::uint32_t(generation));
  }

  // The one pair rule: equal valid labels in one space.
  constexpr bool pair_exact(const label &evidence, const label &reference) {
    return evidence.valid && reference.valid && evidence.space != label_space::none && evidence.space == reference.space &&
           evidence.value == reference.value;
  }

  // Why a pair is not exact: the evidence's refusal, no reference, the
  // reference's refusal, then a mismatch; none when pair_exact.
  constexpr refusal pair_refusal(const label &evidence, const label &reference) {
    if (!evidence.valid) {
      return evidence.reason == refusal::none ? refusal::unstamped : evidence.reason;
    }
    if (reference.space == label_space::none) {
      return refusal::no_reference;
    }
    if (!reference.valid) {
      return reference.reason == refusal::none ? refusal::unstamped : reference.reason;
    }
    return pair_exact(evidence, reference) ? refusal::none : refusal::mismatch;
  }

  // The refusal of a snapshot, a label or a pair, by precedence: capture
  // refusals (no proven snapshot), then identity refusals (no proven label),
  // then pair refusals.
  struct refusal_inputs {
    bool begin_refused {}, budget {}, render_pass {}, recording_not_covered {}, enhanced_barrier {};
    bool stale_scope {}, foreign_queue {}, unstamped {}, not_real_span {};
    bool no_reference {}, mismatch {};
  };

  constexpr refusal first_refusal(const refusal_inputs &in) {
    const std::pair<bool, refusal> order[] {
      {in.begin_refused, refusal::begin_refused},
      {in.budget, refusal::budget},
      {in.render_pass, refusal::render_pass},
      {in.recording_not_covered, refusal::recording_not_covered},
      {in.enhanced_barrier, refusal::enhanced_barrier},
      {in.stale_scope, refusal::stale_scope},
      {in.foreign_queue, refusal::foreign_queue},
      {in.unstamped, refusal::unstamped},
      {in.not_real_span, refusal::not_real_span},
      {in.no_reference, refusal::no_reference},
      {in.mismatch, refusal::mismatch},
    };
    for (const auto &[set, reason] : order) {
      if (set) {
        return reason;
      }
    }
    return refusal::none;
  }

  // ---------------------------------------------------------------- frame-generation interposers

  // Modules whose presence means a frame-generation interposer may present
  // generated frames (frame_clock::loaded_interposers probes them).
  // sl.interposer counts as FG-capable: a Streamline session that does not
  // report its FG state cannot prove Present-time evidence real. FSR3 frame
  // generation linked statically into a game executable is invisible here.
  namespace interposer {
    inline constexpr std::uint32_t sl_interposer = 1, sl_dlss_g = 2, ngx_dlssg = 4, fidelityfx_fg = 8, xess_fg = 16;
    inline constexpr std::array<std::string_view, 5> short_names {"sl_interposer", "sl_dlss_g", "ngx_dlssg", "fidelityfx_fg", "xess_fg"};
  }  // namespace interposer

  struct interposer_module {
    std::wstring_view module;  // Points at a NUL-terminated literal.
    std::uint32_t bit;
  };

  inline constexpr std::array<interposer_module, 9> interposer_modules {{
    {L"sl.interposer.dll", interposer::sl_interposer},
    {L"sl.dlss_g.dll", interposer::sl_dlss_g},
    {L"nvngx_dlssg.dll", interposer::ngx_dlssg},
    {L"amd_fidelityfx_dx12.dll", interposer::fidelityfx_fg},
    {L"amd_fidelityfx_framegeneration_dx12.dll", interposer::fidelityfx_fg},
    {L"amd_fidelityfx_loader_dx12.dll", interposer::fidelityfx_fg},
    {L"ffx_frameinterpolation_x64.dll", interposer::fidelityfx_fg},
    {L"ffx_fsr3_x64.dll", interposer::fidelityfx_fg},
    {L"libxess_fg.dll", interposer::xess_fg},
  }};

  // The interposer bit of a module file name, ASCII case-insensitive; zero
  // for any other name.
  constexpr std::uint32_t interposer_bit(std::wstring_view name) {
    const auto lower = [](wchar_t c) {
      return c >= L'A' && c <= L'Z' ? wchar_t(c - L'A' + L'a') : c;
    };
    for (const auto &entry : interposer_modules) {
      if (entry.module.size() != name.size()) {
        continue;
      }
      bool same = true;
      for (std::size_t i = 0; same && i != name.size(); ++i) {
        same = lower(name[i]) == entry.module[i];
      }
      if (same) {
        return entry.bit;
      }
    }
    return 0u;
  }

  // "none", or the short names of the bits, comma-joined.
  inline std::string interposer_names(std::uint32_t bits) {
    std::string text;
    for (std::size_t i = 0; i != interposer::short_names.size(); ++i) {
      if (!(bits & (1u << i))) {
        continue;
      }
      if (!text.empty()) {
        text += ',';
      }
      text += interposer::short_names[i];
    }
    return text.empty() ? std::string("none") : text;
  }

  // The log text after "Sunshine FG interposers: runtime=<p> ", logged when
  // it changes.
  inline std::string format_interposers(std::uint32_t bits, bool fg_known, bool fg_enabled) {
    return "bits=" + std::to_string(bits) + " names=" + interposer_names(bits) + " fg_known=" + (fg_known ? "1" : "0") +
           " fg_enabled=" + (fg_enabled ? "1" : "0");
  }

  // E2: whether evidence captured at a Present belongs to a real frame. Never
  // while the adapter reports FG on (or requested while suspended); always
  // without an FG-capable module; otherwise only while the Streamline adapter
  // reports FG known and off and no FidelityFX or XeSS frame generation
  // (which it cannot see) is loaded.
  constexpr bool present_time_exact(std::uint32_t bits, bool fg_known, bool fg_enabled) {
    if (fg_known && fg_enabled) {
      return false;
    }
    if (!bits) {
      return true;
    }
    return fg_known && !fg_enabled && !(bits & (interposer::fidelityfx_fg | interposer::xess_fg));
  }

  // A game copy's present label holds only when every Present from the
  // labelled one to this one was real: Present-time evidence is exact now and
  // the run of real Presents with FG known off (fix 3's
  // change_set::next_fg_off_presents, this one included) covers the
  // presents_ago Presents since the labelled one.
  constexpr bool real_span(std::uint32_t fg_off_run, std::uint32_t presents_ago, std::uint32_t interposer_bits, bool fg_known, bool fg_enabled) {
    return present_time_exact(interposer_bits, fg_known, fg_enabled) && fg_off_run > presents_ago;
  }

  // ---------------------------------------------------------------- tickets

  struct ticket {
    capture_kind kind {capture_kind::sl_tag};
    ui_selection::kind source {};
    boundary at {boundary::at_tag};
    label token, present;
    // The state proof, and the executing queue relative to the presenting
    // queue (present labels) and the Backbuffer producer's (token labels).
    proof how {};
    queue_relation present_queue {}, token_queue {};
    refusal refused {};
    begin_stage stage {};  // With refusal::begin_refused.
    // Encoding (A1): the typed view format and the swapchain colour space.
    std::uint32_t typed_format {}, color_space {};
    // Scope, and for Streamline snapshots the probe's token generation
    // (unique per slGetNewFrameToken issue; 0 none).
    std::uint64_t epoch {};
    std::uint32_t viewport {};
    std::uint64_t token_generation {};

    // The A1 acceptance key of the captured source.
    std::string key() const {
      return ui_selection::signature {source, typed_format, color_space}.key();
    }

    bool labelled() const {
      return token.valid || present.valid;
    }
  };

  // Two Streamline snapshots of one game frame: the same scope and viewport
  // and the same nonzero token generation (numbered tokens and unnumbered
  // ones alike: the generation is unique per token issue).
  constexpr bool same_frame(const ticket &a, const ticket &b) {
    return a.epoch == b.epoch && a.viewport == b.viewport && a.token_generation && a.token_generation == b.token_generation;
  }

  // The newest token generation among offered Streamline snapshots with a
  // valid token label (0 when none): the real frame a Present offers (T1).
  inline std::uint64_t newest_token(const std::array<ticket, slot::count> &tickets) {
    std::uint64_t newest = 0;
    for (const auto &t : tickets) {
      if (t.kind == capture_kind::sl_tag && t.token.valid) {
        newest = std::max(newest, t.token_generation);
      }
    }
    return newest;
  }

  // ---------------------------------------------------------------- shadow comparison

  // Today's pairing for one render (Present counting).
  struct today_view {
    bool offered {};  // A HUD-less selection was offered.
    bool detects {};  // The pair was admitted and detects afresh.
    bool holds {};  // The render holds the previous real frame's decision.
    bool exact {};  // hudless_exact.
    bool batch {};  // Paired as one tag batch.
    std::uint64_t real_frame {};
  };

  // What the ticket would decide for the same render (identity_authoritative).
  struct ticket_view {
    std::uint64_t frame_id {};
    bool fresh {};  // The first render of this frame identity.
    bool hold_previous {};  // A later render of the same identity: hold, never detect again.
    bool paired {};  // The ticket's pair is exact.
    bool batch {};
    std::uint64_t real_frame {};
  };

  // Indices of identity_counters::value.
  namespace identity_counter {
    inline constexpr std::size_t renders = 0, tagged = 1;
    // Per frame identity with at least one render offered a HUD-less image:
    // today's fresh detections were exactly one, none or more than one
    // (extra: the detections beyond the first).
    inline constexpr std::size_t frames_total = 2, frames_once = 3, frames_missed = 4, frames_repeated = 5, frames_extra = 6;
    // Per render: detecting afresh, where either does.
    inline constexpr std::size_t detect_agree = 7, detect_today_only = 8, detect_ticket_only = 9;
    // Per render offered a HUD-less image: the pair's exactness (inexact: both inexact).
    inline constexpr std::size_t exact_agree = 10, exact_today_only = 11, exact_ticket_only = 12, exact_inexact = 13;
    // Per render: one tag batch, where either says so.
    inline constexpr std::size_t batch_agree = 14, batch_today_only = 15, batch_ticket_only = 16;
    // Per ticket counted (count_ticket): its valid labels, or none.
    inline constexpr std::size_t label_token = 17, label_present = 18, label_none = 19;
    // Per ticket counted: its refusal (the index of none unused).
    inline constexpr std::size_t refused = 20;
    // The GPU verdicts of proposed pairs (count_gpu_verdict; the renderer's
    // GPU verification fills them), token_exact the exact ones in token space.
    inline constexpr std::size_t gpu_exact = refused + std::size_t(refusal::count), gpu_mismatch = gpu_exact + 1,
                                 gpu_unstamped = gpu_exact + 2, gpu_unproposed = gpu_exact + 3, gpu_token_exact = gpu_exact + 4;
    // The latest interposer bits, assigned rather than added.
    inline constexpr std::size_t interposers = gpu_token_exact + 1;
    inline constexpr std::size_t count = interposers + 1;
  }  // namespace identity_counter

  static_assert(identity_counter::refused == 20 && identity_counter::gpu_exact == 32 && identity_counter::interposers == 37 && identity_counter::count == 38);

  struct identity_counters {
    std::array<std::uint64_t, identity_counter::count> value {};

    std::uint64_t &operator[](std::size_t index) {
      return value[index];
    }

    std::uint64_t operator[](std::size_t index) const {
      return value[index];
    }

    // Adds counts; interposers takes the other's bits when it has any.
    identity_counters &operator+=(const identity_counters &other) {
      for (std::size_t i = 0; i != value.size(); ++i) {
        if (i != identity_counter::interposers) {
          value[i] += other.value[i];
        }
      }
      if (other.value[identity_counter::interposers]) {
        value[identity_counter::interposers] = other.value[identity_counter::interposers];
      }
      return *this;
    }

    // Field-wise difference of two snapshots of the same running totals; the
    // later interposer bits.
    friend identity_counters operator-(const identity_counters &later, const identity_counters &earlier) {
      identity_counters result;
      for (std::size_t i = 0; i != result.value.size(); ++i) {
        result.value[i] = later.value[i] - earlier.value[i];
      }
      result.value[identity_counter::interposers] = later.value[identity_counter::interposers];
      return result;
    }

    friend bool operator==(const identity_counters &a, const identity_counters &b) {
      return a.value == b.value;
    }

    friend bool operator!=(const identity_counters &a, const identity_counters &b) {
      return !(a == b);
    }
  };

  // Counts one ticket: its valid labels (or none) and its refusal.
  inline void count_ticket(const ticket &t, identity_counters &c) {
    namespace n = identity_counter;
    c[n::label_token] += t.token.valid ? 1 : 0;
    c[n::label_present] += t.present.valid ? 1 : 0;
    c[n::label_none] += t.labelled() ? 0 : 1;
    if (t.refused != refusal::none && t.refused < refusal::count) {
      ++c[n::refused + std::size_t(t.refused)];
    }
  }

  // Counts one GPU verdict of a proposed pair; token_space: the pair was in
  // token space.
  inline void count_gpu_verdict(gpu_verdict verdict, bool token_space, identity_counters &c) {
    namespace n = identity_counter;
    switch (verdict) {
      case gpu_verdict::exact:
        ++c[n::gpu_exact];
        c[n::gpu_token_exact] += token_space ? 1 : 0;
        break;
      case gpu_verdict::mismatch:
        ++c[n::gpu_mismatch];
        break;
      case gpu_verdict::unstamped:
        ++c[n::gpu_unstamped];
        break;
      case gpu_verdict::unproposed:
        ++c[n::gpu_unproposed];
        break;
      default:
        break;
    }
  }

  // Compares today's pairing with the ticket's identity, render by render,
  // and counts per frame identity how often today detected afresh. Under
  // DLSS-G the first Present carrying a new token is a generated one, so
  // real and generated Presents are compared per identity, not per Present.
  // One instance per runtime; end_scope when the source's scope changes.
  class identity_shadow {
  public:
    ticket_view step(const today_view &today, std::uint64_t frame_id, bool ticket_pair_exact, bool ticket_batch, identity_counters &c) {
      namespace n = identity_counter;
      ++c[n::renders];
      ticket_view view;
      view.frame_id = view.real_frame = frame_id;
      view.paired = ticket_pair_exact;
      view.batch = ticket_batch;
      if (frame_id) {
        ++c[n::tagged];
        if (frame_id != current_ && frame_id != finalized_) {
          finalize(c);
          current_ = frame_id;
          view.fresh = true;
        }
        view.hold_previous = !view.fresh;
      }
      const bool today_detects = today.detects && !today.holds;
      if (frame_id && frame_id == current_) {
        offered_ = offered_ || today.offered;
        detections_ += today_detects ? 1u : 0u;
      } else if (frame_id && today_detects) {
        // A detection for an identity already finalized (a late render).
        ++c[n::frames_extra];
      }
      const bool ticket_detects = view.fresh && view.paired;
      if (today_detects || ticket_detects) {
        agreement(c, n::detect_agree, today_detects, ticket_detects);
      }
      if (today.offered) {
        if (!today.exact && !ticket_pair_exact) {
          ++c[n::exact_inexact];
        } else {
          agreement(c, n::exact_agree, today.exact, ticket_pair_exact);
        }
      }
      if (today.batch || ticket_batch) {
        agreement(c, n::batch_agree, today.batch, ticket_batch);
      }
      return view;
    }

    // Finalizes the current identity and forgets it (the source's scope changed).
    void end_scope(identity_counters &c) {
      finalize(c);
      finalized_ = 0;
    }

  private:
    // Counts agree, today_only or ticket_only, consecutive from agree; at
    // least one side is true.
    static void agreement(identity_counters &c, std::size_t agree, bool today, bool ticket) {
      std::size_t index = agree;
      if (!today) {
        index += 2;
      } else if (!ticket) {
        index += 1;
      }
      ++c[index];
    }

    void finalize(identity_counters &c) {
      namespace n = identity_counter;
      if (current_ && offered_) {
        ++c[n::frames_total];
        if (!detections_) {
          ++c[n::frames_missed];
        } else if (detections_ == 1) {
          ++c[n::frames_once];
        } else {
          ++c[n::frames_repeated];
          c[n::frames_extra] += detections_ - 1;
        }
      }
      if (current_) {
        finalized_ = current_;
      }
      current_ = 0;
      detections_ = 0;
      offered_ = false;
    }

    std::uint64_t current_ {}, finalized_ {};
    std::uint64_t detections_ {};
    bool offered_ {};
  };

  // The log text after "Sunshine UI identity: runtime=<p> ": space-separated
  // key=value fields, groups as key={key=value ...} (as format_ui_counters);
  // interposers in decimal bits.
  inline std::string format_identity_counters(const identity_counters &c) {
    namespace n = identity_counter;
    std::string text;
    const auto field = [&](std::string_view key, std::uint64_t value) {
      if (!text.empty() && text.back() != '{') {
        text += ' ';
      }
      text.append(key).append("=").append(std::to_string(value));
    };
    const auto group = [&](std::string_view key, auto &&fields) {
      if (!text.empty()) {
        text += ' ';
      }
      text.append(key).append("={");
      fields();
      text += '}';
    };
    field("renders", c[n::renders]);
    field("tagged", c[n::tagged]);
    group("frames", [&] {
      field("total", c[n::frames_total]);
      field("once", c[n::frames_once]);
      field("missed", c[n::frames_missed]);
      field("repeated", c[n::frames_repeated]);
      field("extra", c[n::frames_extra]);
    });
    group("detect", [&] {
      field("agree", c[n::detect_agree]);
      field("today_only", c[n::detect_today_only]);
      field("ticket_only", c[n::detect_ticket_only]);
    });
    group("exact", [&] {
      field("agree", c[n::exact_agree]);
      field("today_only", c[n::exact_today_only]);
      field("ticket_only", c[n::exact_ticket_only]);
      field("inexact", c[n::exact_inexact]);
    });
    group("batch", [&] {
      field("agree", c[n::batch_agree]);
      field("today_only", c[n::batch_today_only]);
      field("ticket_only", c[n::batch_ticket_only]);
    });
    group("label", [&] {
      field("token", c[n::label_token]);
      field("present", c[n::label_present]);
      field("none", c[n::label_none]);
    });
    group("refused", [&] {
      for (std::size_t r = 1; r != refusal_names.size(); ++r) {
        field(refusal_names[r], c[n::refused + r]);
      }
    });
    group("gpu", [&] {
      field("exact", c[n::gpu_exact]);
      field("mismatch", c[n::gpu_mismatch]);
      field("unstamped", c[n::gpu_unstamped]);
      field("unproposed", c[n::gpu_unproposed]);
      field("token_exact", c[n::gpu_token_exact]);
    });
    field("interposers", c[n::interposers]);
    return text;
  }
}  // namespace sunshine_game3d::ui_ticket
