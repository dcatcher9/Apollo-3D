// SPDX-License-Identifier: GPL-3.0-only
#include "streamline_camera_probe.h"
#include "addon_lifetime.h"
#include "addon_config.h"
#include "api_contract.h"
#include "streamline_camera_data.h"
#include "streamline_camera_version.h"
#include "streamline_buffer_contract.h"
#include "streamline_v1_resource_state.h"
#include "upscaler_call_trace.h"
#include "game3d_diagnostic_metadata.h"
#include "game3d_ui_mask.h"
#include "observer_snapshot.h"
#include "streamline_camera_probe_test_access.h"
#include "streamline_native_observer.h"
#include <MinHook.h>
#include <reshade.hpp>
#include "async_log.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#define SUNSHINE_UPSCALER_CALLER reinterpret_cast<std::uintptr_t>(_ReturnAddress())
#else
#define SUNSHINE_UPSCALER_CALLER reinterpret_cast<std::uintptr_t>(__builtin_extract_return_addr(__builtin_return_address(0)))
#endif

namespace sunshine_streamline {
  // A Streamline tag's contract: its state proof, lifetime and the lifecycle
  // match() judges (api_contract.h).
  namespace sl_contract = sunshine_api_contract::streamline;
  SUNSHINE_PROBE_INTERNAL {
    namespace dump_metadata = sunshine_game3d::diagnostic;
    dump_metadata::stamp dump_stamp(std::uint64_t epoch, std::uint64_t sequence, std::uint64_t tick,
        std::uint32_t viewport = UINT32_MAX, std::uint64_t token = 0, std::uint64_t frame = 0,
        bool numeric = false, std::uint64_t command = 0) {
      return {sunshine_game3d::diagnostic_metadata_generation(), epoch, sequence, tick, token, frame, command, viewport, numeric};
    }
    using abi = versioning::abi;
    enum class origin { global, frame, local };
    // sl_core_api.h sets no limit on numTags or numInputs (the contract
    // matrix's S18): a tag array or input list is read tag_chunk elements at
    // a time, on the stack.
    constexpr unsigned slot_count = frame_generation::viewport_slots, tag_chunk = 32, frame_count = 32;
    // Canonical buffer types this observer keeps, depth kinds first. Every
    // tag array below is indexed by position in this list.
    constexpr std::array<std::uint32_t, 9> observed_tag_types{buffers::depth, buffers::high_resolution_depth,
      buffers::linear_depth, buffers::hudless_color, buffers::scaling_input_color, buffers::scaling_output_color,
      buffers::backbuffer, buffers::ui_color_and_alpha, buffers::ui_alpha};
    constexpr unsigned slot_of(std::uint32_t type) {
      for (unsigned i = 0; i != observed_tag_types.size(); ++i) if (observed_tag_types[i] == type) return i;
      return static_cast<unsigned>(observed_tag_types.size());
    }
    static_assert(slot_of(buffers::depth) == 0 && slot_of(buffers::high_resolution_depth) == 1 &&
      slot_of(buffers::linear_depth) == 2, "depth kinds occupy the first three tag slots");
    // Explicit display-resolution depth first, then render depth, then linear distance.
    constexpr std::array<unsigned, 3> depth_priority{slot_of(buffers::high_resolution_depth), slot_of(buffers::depth),
      slot_of(buffers::linear_depth)};
    // The UI kinds, in the order a batch offers them.
    constexpr std::array<unsigned, 4> ui_priority{slot_of(buffers::hudless_color), slot_of(buffers::ui_alpha),
      slot_of(buffers::ui_color_and_alpha), slot_of(buffers::backbuffer)};
    // Defined in sl_observation.inl.
    bool known_renderer_contract(std::uint32_t feature);
    unsigned tag_index(std::uint32_t type);
    struct tag_record {
      depth_tag value;
      std::uint64_t tick{};
      std::uintptr_t token{};
      origin source{};
      bool present{}, supported{};
      std::uint64_t serial{}, loss{};
      frame_identity frame;
      std::uint32_t native_state{0xffffffffu};
      bool state_requires_observation{};
      // Its call supplied a command list (an evaluate input's evaluation always does).
      bool commanded{};
      depth_capture::source_ref direct_source;
      std::uint64_t source_present_generation{};
      double precision_scale{1.0}, precision_bias{};
    };
    struct viewport_record {
      camera_data camera;
      camera_key key;
      decode_status decoded{decode_status::short_buffer};
      std::array<tag_record, observed_tag_types.size()> global_tags{}, local_tags{};
      std::array<tag_record, observed_tag_types.size()> active_tags{};
      // Per depth kind, the viewport's newest global or frame tag call that
      // named a resource. An untag stored after it leaves it in place, so an
      // evaluate that reads that untag judges the call it ended
      // (ended_depth_tag, tag_call_decided).
      std::array<tag_record, depth_priority.size()> depth_calls{};
      frame_generation_snapshot frame_generation;
      frame_identity camera_frame;
      std::uintptr_t camera_token{};
      std::uint64_t camera_tick{}, camera_serial{}, changed{};
      // The viewport's newest frame: its key (frame_key; 0
      // unknown, and a repeated key is the same frame) and the observation
      // sequence of the constants call that began it.
      std::uint64_t frame_key{}, frame_serial{};
      // Observed camera resets of this viewport; each starts a new temporal history.
      std::uint64_t history{};
      // The newest valid camera's depth coefficients and how many consecutive
      // valid cameras of this viewport repeated them exactly.
      camera_validation steady_projection;
      std::uint32_t steady_frames{};
      std::uint32_t viewport{};
      bool used{}, has_camera{};
    };
    // Consecutive identical valid cameras before a viewport's depth
    // coefficients apply to depth without its own frame's camera.
    constexpr std::uint32_t steady_projection_frames = 8;
    struct targets { void *constants{}, *tag{}, *tag_for_frame{}, *evaluate{}, *new_frame_token{}, *get_feature_function{}; };
    struct batch {
      std::array<tag_record, observed_tag_types.size()> tags{}; // Three depth and six independently tagged UI/color kinds.
      unsigned count{};
      std::uint32_t viewport{};
      std::uint64_t observation{};
      bool valid_viewport{};
      std::uint64_t serial{}, loss{}, tick{};
      frame_identity frame;
    };
    struct frame_record {
      frame_identity frame;
      camera_data camera;
      decode_status decoded{decode_status::short_buffer};
      std::array<tag_record, observed_tag_types.size()> tags{};
      std::uint64_t camera_serial{}, camera_tick{}, loss{}, changed{}, history{};
      std::uint32_t viewport{};
      bool used{}, has_camera{};
    };
    struct token_record { frame_identity frame; };

    // POD/atomic storage remains live in the pinned add-on after AddonUninit. In
    // particular there is no destructible C++ mutex or worker used by detours.
    SRWLOCK presentations_lock = SRWLOCK_INIT;
    SRWLOCK poll_lock = SRWLOCK_INIT;
    struct metadata_state {
      viewport_record records[slot_count]{};
      frame_record frames[frame_count]{};
      std::uint64_t epoch{}, observation{}, revision{};
    };
    using metadata_owner = observation::snapshot_owner<metadata_state, 8>;
    metadata_owner metadata;
    struct token_state {
      token_record tokens[frame_count]{};
      std::uint64_t observation{};
    };
    using token_owner = observation::snapshot_owner<token_state, 8>;
    token_owner token_metadata;
    // Defined in sl_observation.inl.
    token_owner::transaction write_tokens_briefly();
    std::atomic<bool> observing{};
    std::atomic<bool> source_requested{};
    std::atomic<std::uint64_t> observation_generation{};
    std::atomic<std::uint64_t> constant_calls{}, tag_calls{}, evaluation_calls{}, dropped{}, invalid{};
    // Depth tags naming no resource (Streamline's untag): a null Resource
    // pointer, or a Resource whose native is null.
    std::atomic<std::uint64_t> depth_untags{};
    // FG tag boundaries that named no capturable depth.
    std::atomic<std::uint64_t> fg_refusals{};
    // Next tick each bounded tag line may be written (5 s apart).
    std::atomic<std::uint64_t> next_compatibility_report{}, next_untag_report{}, next_fg_refusal_report{};
    // The newest FG copy DLSS-G's own NGX evaluation recorded: its scope (the
    // observation epoch and viewport) and tick (observe_frame_generation_inputs).
    // Within the FG scope's second a command-less FG tag boundary of that
    // scope whose tags offer no copy point withdraws nothing: the evaluation,
    // not the tag, is where its depth is copied (Assassin's Creed Shadows
    // 10-23 tags its FG depth without a command list and untags it on every
    // frame; the untag cancels the copy deferred to the frame's Present, R7).
    SRWLOCK evaluation_supply_lock = SRWLOCK_INIT;
    struct { std::uint64_t epoch{}, tick{}; std::uint32_t viewport{}; } evaluation_supply;
    void remember_evaluation_supply(std::uint64_t scope_epoch, std::uint32_t viewport, std::uint64_t tick) {
      AcquireSRWLockExclusive(&evaluation_supply_lock);
      if (tick >= evaluation_supply.tick || scope_epoch != evaluation_supply.epoch || viewport != evaluation_supply.viewport)
        evaluation_supply = {scope_epoch, tick, viewport};
      ReleaseSRWLockExclusive(&evaluation_supply_lock);
    }
    bool evaluation_supplies(std::uint64_t scope_epoch, std::uint32_t viewport) {
      if (!TryAcquireSRWLockShared(&evaluation_supply_lock)) return false; // Busy: withdraw as before.
      const auto supply = evaluation_supply;
      ReleaseSRWLockShared(&evaluation_supply_lock);
      const auto now = GetTickCount64();
      return supply.tick && supply.epoch == scope_epoch && supply.viewport == viewport &&
        sunshine_freshness::ahead_tolerant_le(supply.tick, now, sunshine_freshness::fg_silent_ms);
    }
    // The newest FG tag boundary's outcome per scope: when it recorded a copy
    // and for which tag call, or none when it was refused. A Streamline FG tag
    // that copies keeps precedence over DLSS-G's evaluation of the same frame,
    // which then adds no second copy (observe_frame_generation_inputs,
    // frame_generation::depth_tag_precedence).
    frame_generation::depth_tag_precedence tag_copy;
    // sequence: the tag call's; frame: the frame it named or was associated
    // with (tag_frame).
    void remember_tag_copy(std::uint64_t scope_epoch, std::uint32_t viewport, bool recorded, std::uint64_t sequence = 0,
        const frame_identity &frame = {}) {
      const frame_generation::frame_mark tag{frame_key(frame), sequence};
      const auto now = GetTickCount64();
      AcquireSRWLockExclusive(&evaluation_supply_lock);
      tag_copy.remember(scope_epoch, viewport, recorded, tag, now);
      ReleaseSRWLockExclusive(&evaluation_supply_lock);
    }
    // A frame whose FG tag left it without a copy (refused, or its copy
    // deferred to the frame's Present cancelled, refused or lapsed): the FG
    // scope owns no depth for it (depth_capture::frame_generation_missed), so
    // the viewport's ordinary SR/RR depth may supply it. sequence is the
    // frame's tag call. While DLSS-G's own evaluation supplies the scope's
    // copies (remember_evaluation_supply), that evaluation, not the tag,
    // decides the frame.
    void miss_frame_generation(std::uint64_t scope_epoch, std::uint32_t viewport, std::uint64_t sequence) {
      AcquireSRWLockShared(&evaluation_supply_lock);
      const auto supply = evaluation_supply;
      ReleaseSRWLockShared(&evaluation_supply_lock);
      if (supply.tick && supply.epoch == scope_epoch && supply.viewport == viewport &&
          sunshine_freshness::ahead_tolerant_le(supply.tick, GetTickCount64(), sunshine_freshness::fg_silent_ms)) return;
      depth_capture::frame_generation_missed(scope_epoch, viewport, sequence);
    }
    // A precedence per UI kind (ui_mask::capture_slot,
    // ui_mask::tag_precedence), not keyed to the frame as depth's is, since a
    // kind's snapshots pair exactly only within one origin (docs/reshade-sbs.md,
    // DLSS-G's inputs): the newest Streamline
    // UI tag copy of that kind for the scope within the source age, and the
    // newest refusal of one deferred to its frame's
    // Present. DLSS-G's input of a kind the game's own tags copy adds no second
    // copy (observe_frame_generation_inputs), so one origin feeds UI detection,
    // its exact pairs included; a kind the tags leave out or cannot copy is
    // DLSS-G's. A tag's copy deferred to its frame's Present (present_deferred)
    // counts when it is made, so DLSS-G's evaluation of the next frame stands
    // aside; while it is pending, DLSS-G's evaluation of its own frame stands
    // aside too (ui_tags_supply), unless the kind's last deferred copy was
    // refused within the source age: then the tags defer none of the kind
    // (capture_ui_tag) and DLSS-G's input supplies it until that refusal ages.
    sunshine_game3d::ui_mask::tag_precedence ui_tag_copy;
    void remember_ui_tag_copy(std::uint64_t scope_epoch, std::uint32_t viewport, unsigned slot, bool recorded,
        bool deferred = false) {
      const auto now = GetTickCount64();
      AcquireSRWLockExclusive(&evaluation_supply_lock);
      ui_tag_copy.remember(scope_epoch, viewport, slot, recorded, deferred, now);
      ReleaseSRWLockExclusive(&evaluation_supply_lock);
    }
    void forget_ui_tag_copy(unsigned slot) {
      AcquireSRWLockExclusive(&evaluation_supply_lock);
      ui_tag_copy.forget(slot);
      ReleaseSRWLockExclusive(&evaluation_supply_lock);
    }
    bool ui_deferral_refused(std::uint64_t scope_epoch, std::uint32_t viewport, unsigned slot) {
      const auto now = GetTickCount64();
      AcquireSRWLockShared(&evaluation_supply_lock);
      const bool refused = ui_tag_copy.deferral_refused(scope_epoch, viewport, slot, now);
      ReleaseSRWLockShared(&evaluation_supply_lock);
      return refused;
    }
    // Copies deferred to their frame's Present (owner ruling R2,
    // api_contract.h deferred_copy; the matrix's S6, S11b and U2): per
    // viewport, the newest frame's frame-generation depth candidates (in
    // priority order) and UI tags that their tag call could not copy.
    // present_deferred makes them at that frame's Present (the first after its
    // work was submitted, or under frame generation after DLSS-G evaluated
    // it, while DLSS-G evaluates every frame), or drops them. Guarded by
    // evaluation_supply_lock; deferral_pending counts used entries.
    struct deferred_depth {
      std::array<depth_capture::input, depth_priority.size()> candidates;
      std::size_t count{};
      std::uint64_t superseded{};
    };
    struct deferred_ui {
      bool used{};
      sunshine_game3d::ui_mask::boundary where;
      depth_capture::input input; // Its sequence is its tag call's.
    };
    struct deferred_frame {
      bool used{};
      // executions: the game executions counted at the frame's newest tag
      // call; evaluations: DLSS-G's (fg_evaluation) there and at the
      // viewport's previous deferred frame's.
      std::uint64_t observation{}, epoch{}, serial{}, tick{}, executions{};
      frame_generation::deferral_evaluations::marks evaluations;
      std::uint32_t viewport{};
      // The frame the tag call named (tag_frame); unavailable when none.
      frame_identity frame;
      deferred_depth depth;
      std::array<deferred_ui, 4> ui{}; // By ui_mask::capture_slot.
    };
    std::array<deferred_frame, slot_count> deferred_frames{};
    std::atomic<unsigned> deferral_pending{};
    // Whether DLSS-G's input of this UI kind stands aside for the game's tags
    // (ui_tag_copy): their copy was made within the source age, or one waits
    // for the scope's frame Present (U2, U5), which is that frame's tag copy.
    // Otherwise both would copy the kind, and at that Present DLSS-G's copy of
    // the frame (still pending) and its previous one hold the kind's two
    // snapshots, so the deferred copy finds no reservation and DLSS-G keeps
    // the kind for good. One shared acquisition per kind.
    bool ui_tags_supply(std::uint64_t scope_epoch, std::uint32_t viewport, unsigned slot) {
      if (slot >= ui_tag_copy.copied.size()) return false;
      const bool deferrals = deferral_pending.load(std::memory_order_acquire) != 0;
      const auto now = GetTickCount64();
      AcquireSRWLockShared(&evaluation_supply_lock);
      bool pending = false;
      if (deferrals)
        for (const auto &entry : deferred_frames)
          pending = pending || (entry.used && entry.epoch == scope_epoch && entry.viewport == viewport && entry.ui[slot].used);
      const bool supply = ui_tag_copy.tags_supply(scope_epoch, viewport, slot, pending, now);
      ReleaseSRWLockShared(&evaluation_supply_lock);
      return supply;
    }
    // Whether DLSSG.Depth of the frame DLSS-G evaluates stands aside for that
    // frame's own FG tag copy (tag_copy): recorded for it, or deferred to its
    // Present and waiting there (the viewport's deferral's newest tag call).
    // evaluated: the frame (evaluated_frame).
    bool tags_supply_frame(std::uint64_t scope_epoch, std::uint32_t viewport, const frame_generation::frame_mark &evaluated) {
      const bool deferrals = deferral_pending.load(std::memory_order_acquire) != 0;
      const auto now = GetTickCount64();
      AcquireSRWLockShared(&evaluation_supply_lock);
      frame_generation::frame_mark pending;
      bool waiting = false;
      if (deferrals)
        for (const auto &entry : deferred_frames)
          if (entry.used && entry.depth.count && entry.epoch == scope_epoch && entry.viewport == viewport) {
            pending = {frame_key(entry.frame), entry.serial};
            waiting = true;
          }
      const bool supply = tag_copy.tags_supply(scope_epoch, viewport, evaluated, waiting ? &pending : nullptr, now);
      ReleaseSRWLockShared(&evaluation_supply_lock);
      return supply;
    }
    // DLSS-G's NGX evaluations of a real frame (observe_frame_generation_inputs
    // with an FG scope): their count, and the newest one's scope and tick.
    // DLSS-G evaluates a frame in that frame's Present, so under frame
    // generation a Present after an evaluation that followed a tag call is
    // the tag's frame's (or a later one's); a Present before it is an earlier
    // frame's, paced or generated, while DLSS-G evaluates every frame
    // (frame_generation::deferral_evaluations). Guarded by
    // evaluation_supply_lock.
    frame_generation::deferral_evaluations fg_evaluation;
    // Totals for the bounded deferral line (report_deferrals).
    struct {
      std::atomic<std::uint64_t> depth{}, ui{}, depth_copied{}, ui_copied{}, depth_refused{}, ui_refused{}, cancelled{},
        lapsed{};
      // Presents at which a pending deferral of a frame-generating viewport
      // was not yet its frame's (an earlier frame's Present).
      std::atomic<std::uint64_t> waited{};
      std::atomic<unsigned> refusal{}; // The newest refused depth copy's record_stage.
    } deferrals;
    std::atomic<std::uint64_t> next_deferral_report{}, reported_deferrals{};
    // Totals for the bounded tag-call copy line (count_tag_call_copy): depth
    // copies recorded inside a frame-generation tag call, and those
    // made from the state the call's list recorded where the declaration of
    // the tag's later use named another (S11).
    struct {
      std::atomic<std::uint64_t> copied{}, recorded_over_declared{};
    } tag_call_copies;
    std::atomic<std::uint64_t> next_tag_call_report{};
    void drop_deferred() {
      AcquireSRWLockExclusive(&evaluation_supply_lock);
      for (auto &entry : deferred_frames) entry = {};
      deferral_pending.store(0, std::memory_order_release);
      ReleaseSRWLockExclusive(&evaluation_supply_lock);
    }
    // SDK calls answered with eWarnOutOfVRAM: accepted, counted for the log.
    std::atomic<std::uint64_t> vram_warnings{};
    // One verdict per 2.x call (abi_v2::accepted): the warning is a success.
    bool sdk_accepted(std::int32_t result) {
      if (result == abi_v2::result_warn_out_of_vram) vram_warnings.fetch_add(1, std::memory_order_relaxed);
      return abi_v2::accepted(result);
    }
    std::atomic<std::uint64_t> call_sequence{}, loss_revision{};
    // Lost constants observations (each may have hidden a camera reset) and
    // observation restarts. With a viewport's own resets this alone bounds
    // temporal history (feedback.revision); other transient losses (a busy
    // lock, an SDK failure) cannot hide a reset and leave history intact.
    std::atomic<std::uint64_t> history_loss{};
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
    thread_local bool entry_policy_armed{}, entry_policy_observed{}, entry_path_selected{}, entry_source_admitted{};
#endif
    loss_diagnostics::journal loss_journal;
    thread_local loss_diagnostics::context loss_context;
    // Nested middleware calls restore the outer callback's metadata. This
    // context is diagnostic only and never participates in source admission.
    struct loss_context_scope {
      loss_diagnostics::context previous{loss_context};
      explicit loss_context_scope(std::uint64_t sequence = 0, std::uint32_t viewport = UINT32_MAX,
          std::uint32_t feature = UINT32_MAX) { loss_context = {sequence, viewport, feature}; }
      ~loss_context_scope() { loss_context = previous; }
    };
    std::atomic<std::uint64_t> epoch{};
    std::uint64_t next_discovery{};
    HMODULE addon_module{};
    abi installed_abi{};
    // Header numbering of the hooked interposer; raw tags are canonicalized once.
    buffers::contract installed_contract{};
    bool hooks_installed{}, permanently_rejected{}, reported_missing{}, minhook_initialized{};
    std::atomic<bool> v1_private_state_ready{};
    bool v1_private_state_checked{};
    targets installed_targets{};
    abi_v1::set_constants original_constants_v1{};
    abi_v1::set_tag original_tag_v1{};
    abi_v1::evaluate_feature original_evaluate_v1{};
    abi_v2::set_constants original_constants_v2{};
    abi_v2::set_tag original_tag_v2{};
    abi_v2::set_tag_for_frame original_frame_tag_v2{};
    abi_v2::evaluate_feature original_evaluate_v2{};
    abi_v2::get_new_frame_token original_new_token_v2{};
    abi_v2::get_feature_function original_feature_function_v2{};

    // Surveyed 2.x sl_core_types.h/sl_dlss_g.h prefixes. These are copied,
    // never called as SDK objects or followed past the supported version.
    constexpr guid precision_guid{0x98f6e9ba, 0x8d16, 0x4831, {0xa8,0x02,0x4d,0x3b,0x52,0xff,0x26,0xbf}};
    constexpr guid fg_options_guid{0xfac5f1cb, 0x2dfd, 0x4f36, {0xa1,0xe6,0x3a,0x9e,0x86,0x52,0x56,0xc5}};
    struct precision_info { base_structure base; std::uint32_t formula{}; float bias{}, scale{1}; };
    using fg_options_prefix = abi_v2::fg_options_prefix;
    using fg_set_options = abi_v2::fg_set_options;
    struct fg_target {
      void *entry{};
      HMODULE owner{};
      std::atomic<fg_set_options> original{};
      std::atomic<unsigned> state{}; // 0 wrapper/pending detour, 1 installed, 2 rejected
    };
    std::array<fg_target, 4> fg_targets;
    unsigned fg_target_count{};
    SRWLOCK fg_lock = SRWLOCK_INIT;
    std::atomic<void *> current_fg_target{};
    std::atomic<bool> fg_available{};
    std::atomic<std::uint64_t> fg_loss_revision{};
    thread_local unsigned fg_forwarding_depth{};

    // Defined in sl_observation.inl.
    void release_camera_records();
    bool same(const guid &a, const guid &b);
    bool read_bytes(const void *source, void *destination, std::size_t size);
    template<class T> bool read(const T *source, T &destination);
    void message(const char *text);
    std::uint64_t lose(loss_diagnostics::reason cause, const char *site, std::uint32_t line,
        loss_diagnostics::context details = loss_context);
    void invalid_observation(loss_diagnostics::reason cause, const char *site, std::uint32_t line);
    void dropped_observation(loss_diagnostics::reason cause, const char *site, std::uint32_t line,
        loss_diagnostics::context details = loss_context);
    bool middleware_requested() {
      return !sunshine_addon_lifetime::stopping() && source_requested.load(std::memory_order_acquire);
    }
    unsigned tag_index(std::uint32_t type) {
      // Scaling input/output colors are not observed.
      if (type == buffers::scaling_input_color || type == buffers::scaling_output_color)
        return static_cast<unsigned>(observed_tag_types.size());
      return slot_of(type);
    }
    // Defined in sl_observation.inl.
    std::uint64_t observation_ticket();
    bool current(std::uint64_t ticket);
    metadata_owner::read_pin read_metadata();
    metadata_owner::transaction write_metadata(std::uint64_t ticket, bool briefly = false);
    void metadata_write_lost(const metadata_owner::transaction &update, const char *site,
        std::uint32_t line, loss_diagnostics::context details = loss_context);
    viewport_record &slot(metadata_state &state, std::uint32_t viewport);
    bool same_frame(const frame_identity &a, const frame_identity &b);
    frame_identity numeric_frame(std::uint32_t frame);
    bool token_current(const metadata_state &state, const frame_identity &frame);
    frame_identity tag_frame(const metadata_state &state, const viewport_record &record, const frame_identity &call,
        bool global, std::uint64_t serial);
    frame_generation::frame_mark evaluated_frame(std::uint32_t viewport);
    frame_identity observed_frame(std::uintptr_t address);
    bool viewport_value(const abi_v2::viewport &value, std::uint32_t &output);
    unsigned retain_fg_target(void *entry, std::uint64_t ticket) {
      if (!entry || !TryAcquireSRWLockExclusive(&fg_lock)) { fg_available = false; return UINT32_MAX; }
      if (!current(ticket)) { ReleaseSRWLockExclusive(&fg_lock); return UINT32_MAX; }
      if (current_fg_target.exchange(entry, std::memory_order_acq_rel) != entry) {
        fg_available = false;
        ++fg_loss_revision;
      }
      for (unsigned i = 0; i != fg_target_count; ++i) if (fg_targets[i].entry == entry) {
        fg_available.store(fg_targets[i].state.load(std::memory_order_acquire) != 2, std::memory_order_release);
        ReleaseSRWLockExclusive(&fg_lock); return i;
      }
      MEMORY_BASIC_INFORMATION memory{};
      HMODULE owner{};
      const bool executable = VirtualQuery(entry, &memory, sizeof(memory)) == sizeof(memory) && memory.State == MEM_COMMIT &&
        memory.Type == MEM_IMAGE && (memory.Protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)) &&
        !(memory.Protect & (PAGE_GUARD | PAGE_NOACCESS));
      if (fg_target_count == fg_targets.size() || !executable ||
          !GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, reinterpret_cast<LPCWSTR>(entry), &owner)) {
        fg_available = false; ReleaseSRWLockExclusive(&fg_lock); return UINT32_MAX;
      }
      const auto index = fg_target_count++;
      auto &target = fg_targets[index];
      target.entry = entry; target.owner = owner;
      target.original.store(reinterpret_cast<fg_set_options>(entry), std::memory_order_release);
      target.state.store(0, std::memory_order_release);
      fg_available.store(true, std::memory_order_release);
      ReleaseSRWLockExclusive(&fg_lock);
      return index;
    }
    template<unsigned Index> std::int32_t hook_fg_options(const abi_v2::viewport &viewport, const fg_options_prefix &options) {
      const DWORD incoming = GetLastError();
      const auto ticket = observation_ticket();
      // A getter wrapper may have loaded the native entry just before the
      // deferred detour was enabled. Its nested entry must only forward, not
      // publish the same SDK call twice. No TLS object needs destruction.
      const bool sample = !fg_forwarding_depth && ticket && fg_available.load(std::memory_order_acquire) &&
        current_fg_target.load(std::memory_order_acquire) == fg_targets[Index].entry;
      const auto serial = sample ? ++call_sequence : 0;
      fg_options_prefix copy{};
      std::uint32_t id{};
      const bool valid_viewport = sample && viewport_value(viewport, id);
      const bool valid = valid_viewport && read(&options, copy) &&
        same(copy.base.type, fg_options_guid) && copy.base.version >= 1 &&
        !copy.base.next && copy.mode <= 3 && (!copy.mode || copy.mode == 3 || copy.generated_frames != 0);
      SetLastError(incoming);
      const auto options_loss = fg_loss_revision.load(std::memory_order_acquire);
      ++fg_forwarding_depth;
      const auto result = fg_targets[Index].original.load(std::memory_order_acquire)(viewport, options);
      --fg_forwarding_depth;
      const DWORD outgoing = GetLastError();
      // Only the sampling entry judges (and counts) the result.
      const bool sdk_ok = sample && sdk_accepted(result);
      std::uint64_t retired_epoch{}, turned_on{}, turned_on_epoch{};
      if (sample) dump_metadata::observe_sl_fg(dump_stamp(ticket, serial, GetTickCount64(), valid_viewport ? id : UINT32_MAX), copy.mode, copy.generated_frames, valid, sdk_ok);
      bool changed = false;
      // A game may set its FG options once: a busy writer must not leave the
      // mode unknown until the next call.
      auto update = sample && current(ticket) ? write_metadata(ticket, true) : metadata_owner::transaction{};
      if (update) {
        if (current(ticket)) {
          if (!valid_viewport) {
            for (auto &record : update->records) {
              changed |= record.frame_generation.known;
              record.frame_generation = {};
            }
          } else {
            auto &state = slot(*update, id).frame_generation;
            if (serial >= state.sequence) {
              const bool known = valid && sdk_ok;
              changed = !state.sequence || state.known != known ||
                (known && (state.mode != copy.mode || state.generated_frames != copy.generated_frames));
              // Games repeat their options every frame: FG stays enabled since
              // the call that turned it on in this epoch and options revision.
              const auto enabled_since = state.known && state.enabled && state.epoch == update->epoch &&
                state.loss_revision == options_loss ? state.enabled_tick : 0;
              const auto now = GetTickCount64();
              state = {update->epoch, serial, now, options_loss, id, copy.mode, copy.generated_frames,
                known, known && copy.mode != 0, valid && copy.mode >= 2};
              if (state.enabled) state.enabled_tick = enabled_since ? enabled_since : now;
              if (state.enabled && !enabled_since) { turned_on = now; turned_on_epoch = update->epoch; }
              if (!known || copy.mode == 0) retired_epoch = update->epoch;
            }
          }
        }
        if (!current(ticket)) retired_epoch = turned_on = 0;
        else if (!update.commit()) { ++fg_loss_revision; retired_epoch = turned_on = 0; }
      } else if (sample && current(ticket)) ++fg_loss_revision;
      update.reset(); // No mutation ownership across capture/COM reentry below.
      // FG's first Present moves Present to DLSS-G's queue after this frame's
      // copies on the game's queue were submitted: those are copied out for
      // the source age from here (U-C1b, depth_capture::frame_generation_enabled).
      if (turned_on) depth_capture::frame_generation_enabled(turned_on);
      // FG On ends the viewport's FG role retirement (retire_source).
      if (turned_on) depth_capture::frame_generation_restored(turned_on_epoch, id);
      // Revoke the exact FG scope, including tags recorded while the failed
      // SDK call was in flight. A concurrent new On may lose one capture;
      // a fresh tag restores it without resurrecting pre-loss pixels.
      if (retired_epoch) depth_capture::retire_source(sunshine_scene_depth::provider_kind::streamline,
        retired_epoch, sunshine_scene_depth::encode_source_id(sunshine_scene_depth::fg_depth_key(id)), serial);
      // Only an actual FG-mode/knowledge transition revokes UI scope. Games
      // can call unsupported Off options after every successful HUDless tag;
      // repeated failure is not a new failure of those independent input tags.
      if (sample && changed && current(ticket) && (!valid || !sdk_ok || retired_epoch)) {
        if (valid_viewport) sunshine_game3d::ui_mask::invalidate_scope(epoch, id);
        else sunshine_game3d::ui_mask::invalidate_all();
      }
      if (changed) {
        char text[256]{};
        std::snprintf(text, sizeof(text), "Sunshine Streamline frame generation: viewport=%u mode=%u generated_frames=%u supported=%u success=%u sdk_result=%d",
          id, copy.mode, copy.generated_frames, valid, sdk_ok, static_cast<int>(result));
        message(text);
      }
      SetLastError(outgoing);
      return result;
    }
    constexpr std::array<fg_set_options, 4> fg_hooks{hook_fg_options<0>, hook_fg_options<1>, hook_fg_options<2>, hook_fg_options<3>};
    bool install_fg_hooks() {
      if (!observing.load(std::memory_order_acquire) || !middleware_requested()) return false;
      std::array<unsigned, 4> pending{}; unsigned count{};
      if (!TryAcquireSRWLockShared(&fg_lock)) return false;
      for (unsigned i = 0; i != fg_target_count; ++i)
        if (fg_targets[i].state.load(std::memory_order_acquire) == 0) pending[count++] = i;
      ReleaseSRWLockShared(&fg_lock);
      bool okay = true;
      for (unsigned i = 0; i != count; ++i) {
        const auto index = pending[i]; auto &slot = fg_targets[index];
        dump_metadata::observe_module(slot.owner, "streamline", "DLSS Frame Generation options");
        void *trampoline{};
        // retain_fg_target already holds the target module, and discovery pins
        // this add-on. Hooks/trampolines remain pass-through after shutdown.
        if (MH_CreateHook(slot.entry, reinterpret_cast<void *>(fg_hooks[index]), &trampoline) != MH_OK) {
          slot.state = 2; okay = false;
        } else {
          slot.original.store(reinterpret_cast<fg_set_options>(trampoline), std::memory_order_release);
          if (MH_EnableHook(slot.entry) == MH_OK) {
            slot.state.store(1, std::memory_order_release);
            message("Sunshine Streamline FG options observer: native entry hooked; cached and new function pointers are observed; mode remains game controlled");
          }
          else { slot.state = 2; okay = false; }
        }
        if (current_fg_target.load(std::memory_order_acquire) == slot.entry)
          fg_available.store(slot.state.load(std::memory_order_acquire) == 1, std::memory_order_release);
      }
      return okay;
    }
    void discover_fg_options() {
      const auto ticket = observation_ticket();
      if (!ticket || installed_abi != abi::v2 || !original_feature_function_v2) return;
      // Engines may cache this public function before the first ReShade poll.
      // Resolve its real entry at discovery so those cached pointers also pass
      // through our deferred detour. A lookup never establishes enabled state.
      // Same verdict as every 2.x call (eWarnOutOfVRAM is a success), but the
      // add-on's own lookup is not a game call, so vram_warnings ignores it.
      const DWORD saved = GetLastError();
      void *entry{};
      if (abi_v2::accepted(original_feature_function_v2(1000, "slDLSSGSetOptions", entry)) && current(ticket) && entry)
        retain_fg_target(entry, ticket);
      SetLastError(saved);
    }
    std::int32_t hook_get_feature_function(std::uint32_t feature, const char *function_name, void *&function) {
      const DWORD incoming = GetLastError();
      const auto ticket = observation_ticket();
      // A PCL marker lookup is still a game SDK call whose eWarnOutOfVRAM
      // reply vram_warnings counts; only its marker target is no longer kept.
      constexpr char expected[] = "slPCLSetMarker";
      char copied_name[sizeof(expected)]{};
      const bool pcl_lookup = ticket && feature == abi_v2::feature_pcl &&
        read_bytes(function_name, copied_name, sizeof(copied_name)) && std::memcmp(copied_name, expected, sizeof(expected)) == 0;
      constexpr char fg_expected[] = "slDLSSGSetOptions";
      char copied_fg_name[sizeof(fg_expected)]{};
      const bool fg_wanted = ticket && feature == 1000 &&
        read_bytes(function_name, copied_fg_name, sizeof(copied_fg_name)) && std::memcmp(copied_fg_name, fg_expected, sizeof(fg_expected)) == 0;
      SetLastError(incoming);
      const auto result = original_feature_function_v2(feature, function_name, function);
      const DWORD outgoing = GetLastError();
      const bool sdk_ok = (pcl_lookup || fg_wanted) && sdk_accepted(result);
      if (fg_wanted && current(ticket)) {
        void *entry{};
        if (sdk_ok && read_bytes(&function, &entry, sizeof(entry))) {
          // The SDK helper caches this pointer and may set On only once. A
          // returned wrapper observes that first call without installing an
          // inline hook or suspending renderer threads in this callback. Poll
          // later detours the native entry to catch previously cached pointers.
          const auto index = retain_fg_target(entry, ticket);
          if (index < fg_hooks.size()) function = reinterpret_cast<void *>(fg_hooks[index]);
        }
        else { current_fg_target = nullptr; fg_available = false; }
      }
      SetLastError(outgoing);
      return result;
    }
    // Defined in sl_observation.inl.
    frame_record *frame_slot(metadata_state &state, std::uint32_t viewport, const frame_identity &identity, std::uint64_t serial, bool create);
    const frame_record *find_frame(const metadata_state &state, std::uint32_t viewport, const frame_identity &identity);
    void store_camera(std::uint32_t viewport, const camera_data &camera, decode_status decoded,
        std::uint64_t frame, bool explicit_frame, std::uintptr_t token, std::uint64_t ticket,
        const frame_identity &identity, std::uint64_t serial, std::uint64_t loss, std::uint64_t entry_tick);
    void store_tags(batch &values);
    evaluation_snapshot capture_evaluation(std::uint32_t viewport, std::uint32_t feature,
        const frame_identity &identity, void *commands, std::uint64_t ticket, std::uint64_t serial,
        std::uint64_t loss, const batch *local = nullptr, const metadata_state *selected_state = nullptr,
        bool tag_boundary = false);
    void finish_evaluation(evaluation_snapshot value, bool success, std::uint64_t ticket);
    bool evaluate_input(const void *node);
    tag_refusal decode_precision(const void *extension, double &scale, double &bias, bool input_chain);
    void append_tag(batch &out, const abi_v2::resource_tag &tag, origin source, std::uintptr_t token, bool report_invalid = true);
    batch read_tags(const abi_v2::viewport &viewport, const abi_v2::resource_tag *tags,
        std::uint32_t count, origin source, std::uintptr_t token);

    bool hook_constants_v1(const abi_v1::constants &values, std::uint32_t frame, std::uint32_t viewport) {
      const DWORD incoming = GetLastError();
      const auto ticket = observation_ticket();
      const bool sample = ticket != 0;
      const auto serial = sample ? ++call_sequence : 0;
      const loss_context_scope loss_scope(serial, viewport);
      const auto loss = loss_revision.load(std::memory_order_acquire);
      const auto entry_tick = sample ? GetTickCount64() : 0;
      const auto diagnostic = dump_stamp(ticket, serial, entry_tick, viewport, 0, frame, true);
      camera_data camera;
      decode_status decoded = decode_status::short_buffer;
      if (sample) {
        ++constant_calls;
        abi_v1::constants copy{};
        const bool readable = read(&values, copy);
        if (readable) decoded = decode_v1_constants(&copy, sizeof(copy), camera);
        if (diagnostic.session) dump_metadata::observe_sl_constants(diagnostic, copy, readable, decoded);
      }
      SetLastError(incoming);
      const bool result = original_constants_v1(values, frame, viewport);
      const DWORD outgoing = GetLastError();
      loss_context.sdk_result = result ? 1 : 0; loss_context.has_sdk_result = true;
      if (diagnostic.session) dump_metadata::finish_sl_call(diagnostic, result);
      if (sample && result) store_camera(viewport, camera, decoded, frame, true, 0, ticket, numeric_frame(frame), serial, loss, entry_tick);
      else if (sample && current(ticket)) lose(loss_diagnostics::reason::sdk_failure, __func__, 1743);
      SetLastError(outgoing);
      return result;
    }
    bool hook_tag_v1(const abi_v1::resource *resource, std::uint32_t type, std::uint32_t viewport, const extent *area) {
      const DWORD incoming = GetLastError();
      const auto ticket = observation_ticket();
      const bool sample = ticket != 0;
      batch out;
      out.observation = ticket;
      out.serial = sample ? ++call_sequence : 0;
      const loss_context_scope loss_scope(out.serial, viewport);
      out.loss = loss_revision.load(std::memory_order_acquire);
      out.tick = sample ? GetTickCount64() : 0;
      const auto diagnostic = dump_stamp(ticket, out.serial, out.tick, viewport);
      const bool resource_has_state = installed_abi != abi::legacy_no_state;
      if (sample && diagnostic.session) dump_metadata::observe_sl_tag_v1(diagnostic, resource, type, area, resource_has_state);
      if (sample) {
        ++tag_calls;
        out.viewport = viewport;
        out.valid_viewport = true;
        const auto canonical_type = buffers::canonical(installed_contract, type);
        const unsigned index = tag_index(canonical_type);
        if (index < 6) {
          abi_v1::resource copy{};
          extent copied_area{};
          if (abi_v1::read_resource(resource, resource_has_state, copy, read_bytes) && (!area || read(area, copied_area))) {
            out.count = index + 1;
            auto &tag = out.tags[index];
            tag.present = true;
            tag.supported = copy.ext == nullptr && copy.type == 0;
            tag.value.refusal = copy.ext ? tag_refusal::extension : copy.type ? tag_refusal::resource_type : tag_refusal::none;
            tag.value.native_resource = tag.supported ? reinterpret_cast<std::uintptr_t>(copy.native) : 0;
            tag.value.type = canonical_type;
            tag.value.viewport = viewport;
            tag.value.area = copied_area;
            tag.native_state = copy.state;
            if (tag.supported && index < 3)
              tag.direct_source = depth_capture::retain_source(tag.value.native_resource);
            tag.source_present_generation = depth_capture::source_present_generation(tag.direct_source);
          } else {
            invalid_observation(loss_diagnostics::reason::invalid_input, __func__, 1787);
            out.count = index + 1;
            out.tags[index].present = true;
            out.tags[index].value.type = canonical_type;
            out.tags[index].value.viewport = viewport;
          }
        }
      }
      SetLastError(incoming);
      const bool result = original_tag_v1(resource, type, viewport, area);
      const DWORD outgoing = GetLastError();
      loss_context.sdk_result = result ? 1 : 0; loss_context.has_sdk_result = true;
      if (diagnostic.session) dump_metadata::finish_sl_call(diagnostic, result);
      if (sample && result && out.count) store_tags(out);
      else if (sample && !result && current(ticket)) lose(loss_diagnostics::reason::sdk_failure, __func__, 1801);
      SetLastError(outgoing);
      return result;
    }
    // Defined in sl_nomination.inl.
    bool unframed_tag_source(const evaluation_snapshot &snapshot);
    const char *source_rejection(const evaluation_snapshot &snapshot);
    bool direct_source_admissible(const evaluation_snapshot &snapshot);
    sl_contract::copy_point copy_point(const evaluation_snapshot &snapshot, const evaluated_depth_tag &tag);
    template<class Tag> sl_contract::tag_declaration declaration(const Tag &tag, bool version_one);
    const char *tag_rejection(const evaluation_snapshot &snapshot, const evaluated_depth_tag &tag,
        match_status *matched = nullptr);
    bool direct_tag_admissible(const evaluation_snapshot &snapshot, const evaluated_depth_tag &tag);
    bool depth_untag(const evaluation_snapshot &snapshot);
    sunshine_scene_depth::jitter_offset depth_frame_jitter(const evaluation_snapshot &snapshot,
        const evaluated_depth_tag &tag);
    // A declared Resource size that disagrees with the allocation is ignored
    // when the tag's extent defines the region (depth_capture::source_region);
    // name it once per process.
    void report_declared_size(const evaluated_depth_tag &tag) {
      static std::atomic<bool> reported{};
      const auto &value = tag.value;
      std::uint32_t width{}, height{};
      if (sunshine_api_contract::null_extent(value.area) || reported.load(std::memory_order_relaxed) ||
          !depth_capture::source_extent(tag.direct_source, width, height) ||
          !((value.resource_width && value.resource_width != width) || (value.resource_height && value.resource_height != height)) ||
          reported.exchange(true, std::memory_order_relaxed)) return;
      char text[320]{};
      std::snprintf(text, sizeof(text),
        "Sunshine Streamline depth: declared resource size %ux%u differs from the %ux%u allocation; the tag extent %u,%u %ux%u "
        "defines the region and the declared size is ignored (tag_type=%u viewport=%u; logged once)",
        value.resource_width, value.resource_height, width, height, value.area.left, value.area.top, value.area.width,
        value.area.height, value.type, value.viewport);
      message(text);
    }
    // Defined in sl_nomination.inl.
    sunshine_scene_depth::frame normalize_depth_frame(const evaluation_snapshot &snapshot,
        const evaluated_depth_tag &tag, bool version_one);
    // A viewport's depth route under FG: none (FG not enabled), owned by its
    // live FG scope, or silent (enabled, but no FG copy accepted within a
    // second: depth_capture::frame_generation_scope_live). A busy metadata
    // read keeps an existing publication's FG ownership.
    // disabled (optional): the viewport's FG options were observed off in
    // this epoch (a known mode 0 under the current options revision).
    enum class fg_route { none, owned, silent };
    // Defined in sl_nomination.inl.
    fg_route frame_generation_route(std::uint32_t viewport, bool *disabled = nullptr);
    bool source_path_selected(const evaluation_snapshot &snapshot);
    bool owns_nested_depth_capture(const evaluation_snapshot &snapshot, bool copy_recorded);
    const char *name(frame_identity_kind value) {
      switch (value) {
        case frame_identity_kind::unavailable: return "unavailable";
        case frame_identity_kind::v1_numeric: return "v1_numeric";
        case frame_identity_kind::v2_observed_token: return "v2_observed_token";
        case frame_identity_kind::v2_constants_call: return "v2_constants_call";
      }
      return "unknown";
    }
    // Why an FG tag boundary named no capturable depth, at most every 5 s.
    // The tag described is the call's first present depth tag in priority
    // order; reason is its first failing gate. A live FG scope is withdrawn,
    // unless DLSS-G's own evaluation supplies its copies (dlssg_supplied); a
    // silent one has no FG capture to withdraw. Either way the frame has no
    // FG copy, so the scope owns no depth for it (miss_frame_generation).
    // Diagnostic only.
    void report_fg_refusal(const evaluation_snapshot &snapshot, const char *reason, bool withdrawn, bool live,
        bool supplied = false) {
      const auto count = fg_refusals.fetch_add(1, std::memory_order_relaxed) + 1;
      const auto now = GetTickCount64();
      auto next = next_fg_refusal_report.load(std::memory_order_relaxed);
      if (now < next || !next_fg_refusal_report.compare_exchange_strong(next, now + 5000, std::memory_order_relaxed)) return;
      static const evaluated_depth_tag none;
      const auto *tag = &none;
      for (const unsigned index : depth_priority) if (snapshot.tags[index].present) { tag = &snapshot.tags[index]; break; }
      char text[720]{};
      std::snprintf(text, sizeof(text),
        "Sunshine Streamline FG depth refused: reason=%s viewport=%u tag_type=%u scope=%s lifecycle=%u native=0x%llx "
        "command=0x%llx frame=%s status=%s fg=%u fg_scope=%s withdrawn=%u dlssg_supplied=%u refusals=%llu; this FG tag call "
        "names no capturable depth; %s",
        reason, snapshot.viewport, tag->present ? tag->value.type : UINT32_MAX,
        !tag->present ? "absent" : tag->scope == tag_scope::active_global ? "global" :
          tag->scope == tag_scope::explicit_frame ? "frame" : "local",
        tag->present ? tag->value.lifecycle : UINT32_MAX, static_cast<unsigned long long>(tag->value.native_resource),
        static_cast<unsigned long long>(snapshot.command_buffer), name(snapshot.frame.kind), name(snapshot.status),
        unsigned(snapshot.frame_generation_input), live ? "live" : "silent", unsigned(withdrawn), unsigned(supplied),
        static_cast<unsigned long long>(count),
        withdrawn ? "the viewport's FG depth source is withdrawn, and the FG scope owns no depth for this frame, so the "
          "viewport's ordinary SR/RR depth may supply it" :
          supplied ? "DLSS-G's own NGX evaluation copies this FG scope's depth (DLSSG.Depth), so nothing is withdrawn" :
          "the FG scope owns no depth for this frame, so nothing is withdrawn and the viewport's ordinary SR/RR depth "
          "may supply it");
      message(text);
    }
    // A depth copy recorded inside a frame-generation tag call (diagnostic:
    // the recorded candidate's): counted, and with it a copy made from the
    // state the call's list recorded over a declaration of the tag's later
    // use that named another (S11). The bounded line comes at most every 5 s,
    // only after such a copy.
    void count_tag_call_copy(const depth_capture::input *candidates, std::size_t count,
        const depth_capture::record_diagnostic &diagnostic) {
      tag_call_copies.copied.fetch_add(1, std::memory_order_relaxed);
      bool over = false;
      for (std::size_t i = 0; i != count && !over; ++i)
        over = candidates[i].resource.native == diagnostic.resource &&
          candidates[i].proof == sunshine_scene_depth::state_proof::declared_for_use && diagnostic.used_observed_state;
      if (!over) return;
      const auto over_total = tag_call_copies.recorded_over_declared.fetch_add(1, std::memory_order_relaxed) + 1;
      const auto now = GetTickCount64();
      auto next = next_tag_call_report.load(std::memory_order_relaxed);
      if (now < next || !next_tag_call_report.compare_exchange_strong(next, now + 5000, std::memory_order_relaxed)) return;
      char text[480]{};
      std::snprintf(text, sizeof(text),
        "Sunshine Streamline FG tag-call copies: copied=%llu recorded_over_declared=%llu state=0x%x declared=0x%x; a "
        "ValidUntilPresent or ValidUntilEvaluate declaration describes the tag's later use, so a tag call copies its "
        "tag from the state the call's list records",
        static_cast<unsigned long long>(tag_call_copies.copied.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(over_total), diagnostic.copy_state, diagnostic.native_state);
      message(text);
    }
    // Defined in sl_nomination.inl.
    const char *candidate_rejection(const evaluation_snapshot &snapshot);
    bool tag_call_decided(const evaluation_snapshot &snapshot, bool version_one);
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
    // The nomination tap (testing::latest_nomination). Only the thread that
    // armed it records, so an unarmed fixture runs exactly as before. Each
    // nominate_depth_source call evaluates its gates as it begins (pending),
    // and its finish_capture keeps the result, while the observation is
    // current, as its viewport's newest nomination, never over a newer one.
    struct tapped_nomination {
      sunshine_streamline::testing::nomination value;
      std::uint64_t epoch{}, observation{};
      bool used{};
    };
    thread_local bool nomination_tap_armed{};
    bool nomination_tap_pending{};
    sunshine_streamline::testing::nomination nomination_tap_value;
    std::array<tapped_nomination, 8> nomination_tap{};
    bool nomination_tap_current(const tapped_nomination &entry) {
      return entry.used && entry.epoch == epoch.load(std::memory_order_acquire) &&
        entry.observation == observation_generation.load(std::memory_order_acquire);
    }
    void tap_nomination(const evaluation_snapshot &snapshot, bool version_one) {
      auto &out = nomination_tap_value;
      out = {};
      out.path_selected = source_path_selected(snapshot);
      out.source_admissible = direct_source_admissible(snapshot);
      for (unsigned i = 0; i != snapshot.tags.size(); ++i) {
        out.tag_admissible[i] = direct_tag_admissible(snapshot, snapshot.tags[i]);
        if (out.path_selected && out.source_admissible && out.tag_admissible[i])
          out.frames[i] = normalize_depth_frame(snapshot, snapshot.tags[i], version_one);
      }
      out.snapshot = snapshot;
      for (auto &tag : out.snapshot.tags) { tag.direct_source.reset(); tag.ended.direct_source.reset(); }
      nomination_tap_pending = true;
    }
    void finish_tap(bool successful, std::uint64_t ticket) {
      if (!nomination_tap_pending) return;
      nomination_tap_pending = false;
      if (!current(ticket)) return;
      auto &value = nomination_tap_value;
      value.successful = successful;
      tapped_nomination *chosen = nullptr;
      for (auto &entry : nomination_tap) if (entry.used && entry.value.snapshot.viewport == value.snapshot.viewport) {
        chosen = &entry; break;
      }
      if (chosen && nomination_tap_current(*chosen) && chosen->value.snapshot.sequence >= value.snapshot.sequence) return;
      for (auto &entry : nomination_tap) if (!chosen && !nomination_tap_current(entry)) chosen = &entry;
      if (!chosen) {
        chosen = &nomination_tap[0];
        for (auto &entry : nomination_tap) if (entry.value.snapshot.sequence < chosen->value.snapshot.sequence) chosen = &entry;
      }
      *chosen = {std::move(value), epoch.load(std::memory_order_acquire), observation_generation.load(std::memory_order_acquire), true};
    }
#endif
    // Defined in sl_nomination.inl.
    std::uint64_t nominate_depth_source(const evaluation_snapshot &snapshot, bool version_one, bool *copy_recorded = nullptr,
        deferred_depth *deferred = nullptr);
    void finish_capture(std::uint64_t captured, bool successful, std::uint64_t ticket);
    bool hook_evaluate_v1(void *commands, std::uint32_t feature, std::uint32_t frame, std::uint32_t viewport);
    // Defined in sl_observation.inl.
    std::int32_t hook_constants_v2(const abi_v2::constants &values, const abi_v2::frame_token &frame, const abi_v2::viewport &viewport);
    // later: where a ValidUntilPresent UI tag is kept to be copied before its
    // frame's Present (R2, the matrix's U2 and U5).
    sunshine_game3d::ui_mask::attempt capture_ui_tag(const batch &out, void *commands,
        const tag_record &tag, std::uint64_t source_epoch, bool fg_enabled, deferred_ui *later = nullptr) {
      namespace mask = sunshine_game3d::ui_mask;
      depth_capture::input input;
      input.epoch = source_epoch; input.sequence = out.serial; input.tick = out.tick;
      input.viewport = out.viewport; input.observation_revision = out.loss;
      input.source_id = sunshine_scene_depth::encode_source_id(sunshine_scene_depth::fg_depth_key(out.viewport));
      input.frame_generation_input = fg_enabled;
      input.source_frame_generation = out.frame.generation;
      input.source_frame_token = tag.token;
      input.source_frame_numeric = out.frame.numeric;
      input.source_frame_has_numeric = out.frame.has_numeric;
      input.source_frame_explicit = tag.source == origin::frame;
      input.resource.native = tag.supported ? tag.value.native_resource : 0;
      input.resource.width = tag.value.resource_width; input.resource.height = tag.value.resource_height;
      input.resource.native_format = tag.value.native_format;
      input.resource.area = {tag.value.area.left, tag.value.area.top, tag.value.area.width, tag.value.area.height};
      // An undeclared state (UINT_MAX) is no claim: only this recording's
      // observation can supply it, under either state policy. A UI tag is a
      // 2.x tag, copied inside its own tag call.
      sunshine_api_contract::apply(sl_contract::map(declaration(tag, false), sl_contract::copy_point::tag_call), input);
      input.force_snapshot = true;
      depth_capture::record_diagnostic retained;
      input.source = depth_capture::retain_source(input.resource.native, &retained);
      input.source_present_generation = depth_capture::source_present_generation(input.source);
      // Validate actual allocation size before recording, even if SL omitted
      // the optional resource dimensions. Partial output extents are rejected.
      if (input.source) { input.resource.width = retained.width; input.resource.height = retained.height; }
      mask::boundary where;
      where.source = input; where.command = reinterpret_cast<std::uint64_t>(commands);
      where.tag_scope = tag.source == origin::frame ? 1u : tag.source == origin::local ? 3u : 0u;
      where.kind = static_cast<mask::source_kind>(tag.value.type);
      where.source_present_generation = input.source_present_generation;
      where.format = input.source ? retained.format : 0;
      where.refusal = tag.value.refusal;
      if (later && input.source && tag.supported) {
        // A tag call copies its UI tag where it copies a frame-generation
        // depth tag (deferred_copy, the matrix's U5 and U2): inside the call
        // from the state its list records, which a declaration of the tag's
        // later use does not refuse (tag_call_proof, S11), else a
        // ValidUntilPresent tag (no list, no recorded state, or a write
        // state) before its frame's Present; an UntilEvaluate tag whose list
        // records none is refused at the call (missing_state), since no
        // evaluate copies a UI tag.
        bool writing = false;
        const bool recorded = commands &&
          depth_capture::recorded_state(reinterpret_cast<std::uint64_t>(commands), input.source, &writing);
        if (sl_contract::deferred_copy(tag.value.lifecycle, commands != nullptr, recorded, writing) ==
            sl_contract::tag_call_copy::present) {
          // Its kind's precedence over DLSS-G's input stays as it is until the
          // copy is made or dropped (present_deferred). While the kind's last
          // deferred copy was refused within the source age, DLSS-G's input
          // supplies it and no copy is deferred: one would only be refused
          // again, refreshing that refusal every frame.
          if (!ui_deferral_refused(source_epoch, out.viewport, mask::capture_slot(where.kind))) {
            later->used = true; later->where = where; later->input = input;
          }
          return {};
        }
        input.proof = sl_contract::tag_call_proof(tag.value.lifecycle, input.proof);
        where.source = input;
      }
      auto attempt = mask::begin(where, input);
      // A recorded copy keeps its kind's precedence over DLSS-G's input of
      // the same kind (ui_tag_copy); a refusal hands that kind over at once.
      if (mask::source_mask(where.kind))
        remember_ui_tag_copy(source_epoch, out.viewport, mask::capture_slot(where.kind), static_cast<bool>(attempt.ticket));
      return attempt;
    }
    using alpha_attempts = std::array<sunshine_game3d::ui_mask::attempt, 4>;
    using deferred_uis = std::array<deferred_ui, 4>;
    // Whether a batch carries a UI input.
    bool carries_ui(const batch &out) {
      return std::any_of(ui_priority.begin(), ui_priority.end(), [&](unsigned index) { return out.tags[index].present; });
    }
    // The UI inputs of a tag call, or an evaluate's local ones (owner ruling
    // R4, the matrix's U3), each copied on that call's list.
    alpha_attempts capture_ui_inputs(const batch &out, void *commands, deferred_uis *later = nullptr) {
      namespace mask = sunshine_game3d::ui_mask;
      const auto source_epoch = epoch.load(std::memory_order_acquire);
      mask::capture_gate_observation gate;
      gate.epoch = source_epoch; gate.revision = out.loss; gate.sequence = out.serial; gate.tick = out.tick;
      gate.viewport = out.valid_viewport ? out.viewport : UINT32_MAX;
      for (const auto index : ui_priority) if (out.tags[index].present)
        gate.seen_kinds |= mask::source_mask(static_cast<mask::source_kind>(out.tags[index].value.type));
      const bool interested = mask::interested(source_epoch, out.loss, out.viewport, out.tick, &gate.matching_requests);
      if (!out.valid_viewport) gate.state = mask::capture_gate::invalid_viewport;
      else if (!current(out.observation)) gate.state = mask::capture_gate::inactive_observation;
      else if (out.loss != loss_revision.load(std::memory_order_acquire)) gate.state = mask::capture_gate::observation_changed;
      else if (!interested) gate.state = gate.matching_requests ? mask::capture_gate::ambiguous_request : mask::capture_gate::no_matching_request;
      else gate.state = gate.seen_kinds ? mask::capture_gate::admitted : mask::capture_gate::no_ui_tags;
      mask::observe_gate(gate);
      if (gate.state != mask::capture_gate::admitted) return {};
      frame_generation_snapshot fg;
      const bool fg_enabled = query_frame_generation(out.viewport, fg) && fg.enabled;
      // Every source uses its own synchronous command/state/lifetime boundary.
      // Capture availability is independent of its eventual per-frame mask
      // validation, and HUD-less color remains a separate scene-color input.
      alpha_attempts attempts;
      for (unsigned i = 0; i != ui_priority.size(); ++i) {
        const auto &tag = out.tags[ui_priority[i]];
        if (tag.present) attempts[i] = capture_ui_tag(out, commands, tag, source_epoch, fg_enabled,
          later ? &(*later)[mask::capture_slot(static_cast<mask::source_kind>(tag.value.type))] : nullptr);
      }
      return attempts;
    }
    // After the SDK call that supplied them: kept only when that call
    // succeeded within the observation the copies were recorded in.
    void finish_ui_inputs(const alpha_attempts &attempts, bool kept) {
      for (const auto &attempt : attempts) {
        sunshine_game3d::ui_mask::finish(attempt, kept);
        // A revoked copy no longer keeps its kind's precedence (ui_tag_copy).
        if (!kept && attempt.ticket) forget_ui_tag_copy(sunshine_game3d::ui_mask::capture_slot(attempt.kind));
      }
    }
    // Defined in sl_observation.inl.
    evaluation_snapshot capture_depth_tag(const batch &out, void *commands);
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
    // The deferral line's totals of depth copies deferred to their frame's
    // Present (R2), of those copied, and of the deferrals an untag cancelled
    // (R7): the session timelines judge each frame's own untag by them.
    extern "C" __declspec(dllexport) void SunshineStreamlineTestDeferrals(std::uint64_t *depth_deferred, std::uint64_t *depth_copied, std::uint64_t *cancelled) {
      if (depth_deferred) *depth_deferred = deferrals.depth.load(std::memory_order_relaxed);
      if (depth_copied) *depth_copied = deferrals.depth_copied.load(std::memory_order_relaxed);
      if (cancelled) *cancelled = deferrals.cancelled.load(std::memory_order_relaxed);
    }
#endif
    // The bounded deferral line, at most every 5 s and only after its totals
    // changed: what the copies deferred to their frame's Present (R2) became.
    void report_deferrals() {
      const auto activity = deferrals.depth.load(std::memory_order_relaxed) + deferrals.ui.load(std::memory_order_relaxed) +
        deferrals.cancelled.load(std::memory_order_relaxed) + deferrals.lapsed.load(std::memory_order_relaxed);
      const auto now = GetTickCount64();
      auto next = next_deferral_report.load(std::memory_order_relaxed);
      if (activity == reported_deferrals.load(std::memory_order_relaxed) || now < next ||
          !next_deferral_report.compare_exchange_strong(next, now + 5000, std::memory_order_relaxed)) return;
      reported_deferrals.store(activity, std::memory_order_relaxed);
      char text[640]{};
      std::snprintf(text, sizeof(text),
        "Sunshine Streamline deferred tag copies: depth={deferred=%llu copied=%llu refused=%llu last_refusal=%s} "
        "ui={deferred=%llu copied=%llu refused=%llu} cancelled_by_untag=%llu lapsed=%llu waited_for_frame=%llu "
        "pending=%u; a ValidUntilPresent tag whose call has no command list, or a frame-generation or UI input whose "
        "call's list records no state of it or a write state, is copied on the Present queue right before its frame's "
        "Present",
        static_cast<unsigned long long>(deferrals.depth.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(deferrals.depth_copied.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(deferrals.depth_refused.load(std::memory_order_relaxed)),
        depth_capture::name(static_cast<depth_capture::record_stage>(deferrals.refusal.load(std::memory_order_relaxed))),
        static_cast<unsigned long long>(deferrals.ui.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(deferrals.ui_copied.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(deferrals.ui_refused.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(deferrals.cancelled.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(deferrals.lapsed.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(deferrals.waited.load(std::memory_order_relaxed)),
        deferral_pending.load(std::memory_order_relaxed));
      message(text);
    }
    // Requires evaluation_supply_lock. Recounts the used deferral entries.
    void count_deferred() {
      unsigned used = 0;
      for (const auto &entry : deferred_frames) used += entry.used;
      deferral_pending.store(used, std::memory_order_release);
    }
    // The kinds a deferral entry holds, for the precedence handed back when
    // they are dropped: its depth (bit 0) and its UI capture slots (bit 1 +
    // slot).
    unsigned deferred_kinds(const deferred_frame &entry) {
      unsigned kinds = entry.depth.count ? 1u : 0u;
      for (unsigned slot = 0; slot != entry.ui.size(); ++slot) if (entry.ui[slot].used) kinds |= 2u << slot;
      return kinds;
    }
    unsigned kind_count(unsigned kinds) {
      unsigned count = 0;
      for (; kinds; kinds &= kinds - 1) ++count;
      return count;
    }
    // A dropped deferral no longer keeps its kinds' precedence over DLSS-G's
    // own inputs (tag_copy, ui_tag_copy): this frame has no tag copy, so its
    // FG scope owns no depth for it either (miss_frame_generation). sequence:
    // the frame's newest tag call.
    void hand_over(std::uint64_t scope_epoch, std::uint32_t viewport, unsigned kinds, std::uint64_t sequence) {
      if (kinds & 1u) {
        remember_tag_copy(scope_epoch, viewport, false);
        miss_frame_generation(scope_epoch, viewport, sequence);
      }
      for (unsigned slot = 0; slot != 4; ++slot)
        if (kinds & (2u << slot)) remember_ui_tag_copy(scope_epoch, viewport, slot, false);
    }
    // After a 2.x tag call the SDK accepted: keep its copies deferred to its
    // frame's Present (R2), and let its untags cancel the viewport's deferred
    // copies of their frame (R7, api_contract.h untag_cancels). tokened: the
    // call named a frame token (slSetTagForFrame). ui_current: its UI inputs
    // were admitted under the current observation.
    void defer_tag_call(const batch &out, bool tokened, deferred_depth &depth, deferred_uis &ui, bool ui_current) {
      namespace mask = sunshine_game3d::ui_mask;
      if (!ui_current) for (auto &value : ui) value = {};
      const bool keeps_ui = std::any_of(ui.begin(), ui.end(), [](const deferred_ui &value) { return value.used; });
      // An untag: the call's depth tags all name no native (as depth_untag), or
      // a UI kind's tag names none.
      bool untags_depth = false, depth_named = false;
      for (unsigned i = 0; i != 3; ++i) if (out.tags[i].present) {
        if (out.tags[i].supported && !out.tags[i].value.native_resource) untags_depth = true;
        else depth_named = true;
      }
      untags_depth = untags_depth && !depth_named;
      unsigned ui_untags = 0;
      for (unsigned i = 3; i != out.tags.size(); ++i) {
        const auto &tag = out.tags[i];
        const auto kind = static_cast<mask::source_kind>(tag.value.type);
        if (tag.present && tag.supported && !tag.value.native_resource && mask::source_mask(kind))
          ui_untags |= 2u << mask::capture_slot(kind);
      }
      if (!depth.count && !keeps_ui && !untags_depth && !ui_untags) return;
      // The game's work executed before the frame's Present proves the
      // sources' states there (the declaration must agree).
      for (std::size_t i = 0; i != depth.count; ++i) depth_capture::track_deferred_source(depth.candidates[i].source);
      for (const auto &item : ui) if (item.used) depth_capture::track_deferred_source(item.input.source);
      frame_identity frame;
      if (auto pin = read_metadata()) for (const auto &record : pin->records)
        if (record.used && record.viewport == out.viewport) { frame = tag_frame(*pin, record, out.frame, !tokened, out.serial); break; }
      const bool framed = frame.kind != frame_identity_kind::unavailable;
      const auto source_epoch = epoch.load(std::memory_order_acquire);
      const auto executions = depth_capture::game_executions();
      unsigned cancelled = 0, dropped = 0;
      std::uint64_t dropped_epoch{}, cancelled_serial{}, dropped_serial{};
      std::uint32_t dropped_viewport{};
      AcquireSRWLockExclusive(&evaluation_supply_lock);
      deferred_frame *entry = nullptr;
      for (auto &value : deferred_frames) if (value.used && value.viewport == out.viewport) { entry = &value; break; }
      if (entry && (untags_depth || ui_untags) &&
          sl_contract::untag_cancels(tokened && framed, entry->frame.kind != frame_identity_kind::unavailable,
            same_frame(frame, entry->frame))) {
        cancelled = deferred_kinds(*entry) & ((untags_depth ? 1u : 0u) | ui_untags);
        cancelled_serial = entry->serial;
        if (cancelled & 1u) entry->depth = {};
        for (unsigned slot = 0; slot != entry->ui.size(); ++slot) if (cancelled & (2u << slot)) entry->ui[slot] = {};
      }
      if (depth.count || keeps_ui) {
        bool begins = false;
        // Another frame's deferral of the viewport lapses: its Present is past.
        if (entry && (entry->epoch != source_epoch || entry->observation != out.observation ||
            !(same_frame(entry->frame, frame) || (!framed && entry->frame.kind == frame_identity_kind::unavailable)))) {
          dropped = deferred_kinds(*entry); dropped_epoch = entry->epoch; dropped_viewport = entry->viewport;
          dropped_serial = entry->serial;
          *entry = {};
          entry = nullptr;
        }
        if (!entry) {
          // A free entry, or the viewport entry tagged longest ago (lapsed).
          for (auto &value : deferred_frames) if (!value.used) { entry = &value; break; }
          if (!entry) {
            entry = &deferred_frames[0];
            for (auto &value : deferred_frames) if (value.serial < entry->serial) entry = &value;
            dropped |= deferred_kinds(*entry); dropped_epoch = entry->epoch; dropped_viewport = entry->viewport;
            dropped_serial = entry->serial;
            *entry = {};
          }
          entry->used = true; entry->observation = out.observation; entry->epoch = source_epoch;
          entry->viewport = out.viewport; entry->frame = frame;
          begins = true;
        }
        // The frame's newest tag call: its submitted work (and under frame
        // generation DLSS-G's evaluation of it, while DLSS-G evaluated the
        // viewport's previous deferred frame) must follow it.
        entry->serial = out.serial; entry->tick = out.tick; entry->executions = executions;
        entry->evaluations = fg_evaluation.tag(source_epoch, out.viewport, out.serial, begins, entry->evaluations,
          depth_capture::next_present_ordinal());
        if (depth.count) { entry->depth = std::move(depth); deferrals.depth.fetch_add(1, std::memory_order_relaxed); }
        for (unsigned slot = 0; slot != ui.size(); ++slot) if (ui[slot].used) {
          entry->ui[slot] = std::move(ui[slot]);
          deferrals.ui.fetch_add(1, std::memory_order_relaxed);
        }
      }
      if (entry && !deferred_kinds(*entry)) *entry = {};
      count_deferred();
      ReleaseSRWLockExclusive(&evaluation_supply_lock);
      if (cancelled) {
        deferrals.cancelled.fetch_add(kind_count(cancelled), std::memory_order_relaxed);
        hand_over(source_epoch, out.viewport, cancelled, cancelled_serial);
      }
      if (dropped) {
        deferrals.lapsed.fetch_add(kind_count(dropped), std::memory_order_relaxed);
        hand_over(dropped_epoch, dropped_viewport, dropped, dropped_serial);
      }
      report_deferrals();
    }
    // Defined in sl_observation.inl.
    std::int32_t hook_tag_v2(const abi_v2::viewport &viewport, const abi_v2::resource_tag *tags, std::uint32_t count, void *commands);
    std::int32_t hook_frame_tag_v2(const abi_v2::frame_token &frame, const abi_v2::viewport &viewport,
        const abi_v2::resource_tag *tags, std::uint32_t count, void *commands);
    batch read_local_inputs(const base_structure **inputs, std::uint32_t count, std::uintptr_t token, bool report_invalid);
    // Defined in sl_nomination.inl.
    std::int32_t hook_evaluate_v2(std::uint32_t feature, const abi_v2::frame_token &frame,
        const base_structure **inputs, std::uint32_t count, void *commands);

    // Defined in sl_observation.inl.
    std::int32_t hook_new_token_v2(abi_v2::frame_token *&token, const std::uint32_t *frame_index);

    bool install_hooks(abi version, const targets &functions, const buffers::contract &contract) {
      if (version == abi::unsupported || !functions.constants || !functions.tag || !functions.evaluate) return false;
      if (!contract.known) return false;
      if (!minhook_initialized) {
        const auto status = MH_Initialize();
        if (status != MH_OK && status != MH_ERROR_ALREADY_INITIALIZED) return false;
        minhook_initialized = true;
      }
      struct entry { void *target, *hook; void **original; };
      std::array<entry, 6> entries{};
      unsigned count{};
      if (versioning::is_legacy(version)) {
        entries[count++] = {functions.constants, reinterpret_cast<void *>(hook_constants_v1), reinterpret_cast<void **>(&original_constants_v1)};
        entries[count++] = {functions.tag, reinterpret_cast<void *>(hook_tag_v1), reinterpret_cast<void **>(&original_tag_v1)};
        entries[count++] = {functions.evaluate, reinterpret_cast<void *>(hook_evaluate_v1), reinterpret_cast<void **>(&original_evaluate_v1)};
      } else {
        entries[count++] = {functions.constants, reinterpret_cast<void *>(hook_constants_v2), reinterpret_cast<void **>(&original_constants_v2)};
        entries[count++] = {functions.tag, reinterpret_cast<void *>(hook_tag_v2), reinterpret_cast<void **>(&original_tag_v2)};
        if (functions.tag_for_frame) entries[count++] = {functions.tag_for_frame, reinterpret_cast<void *>(hook_frame_tag_v2), reinterpret_cast<void **>(&original_frame_tag_v2)};
        entries[count++] = {functions.evaluate, reinterpret_cast<void *>(hook_evaluate_v2), reinterpret_cast<void **>(&original_evaluate_v2)};
        if (functions.new_frame_token) entries[count++] = {functions.new_frame_token, reinterpret_cast<void *>(hook_new_token_v2), reinterpret_cast<void **>(&original_new_token_v2)};
        if (functions.get_feature_function) entries[count++] = {functions.get_feature_function, reinterpret_cast<void *>(hook_get_feature_function), reinterpret_cast<void **>(&original_feature_function_v2)};
      }
      unsigned made = 0;
      for (; made < count; ++made) {
        if (MH_CreateHook(entries[made].target, entries[made].hook, entries[made].original) != MH_OK) break;
      }
      if (made != count) {
        // None was enabled, so these trampolines cannot be running.
        for (unsigned i = 0; i < made; ++i) MH_RemoveHook(entries[i].target);
        return false;
      }
      // Detours canonicalize raw tags from their first call.
      installed_contract = contract;
      buffers::set_active_contract(contract);
      for (unsigned i = 0; i < count; ++i) {
        if (MH_QueueEnableHook(entries[i].target) != MH_OK) return false;
      }
      // On a partial enable failure retain all trampolines in pinned code and
      // disable observation; do not assume no thread entered a detour.
      if (MH_ApplyQueued() != MH_OK) return false;
      installed_targets = functions;
      installed_abi = version;
      hooks_installed = true;
      return true;
    }
    versioning::information module_version(HMODULE module) {
      wchar_t path[32768]{};
      const auto size = GetModuleFileNameW(module, path, static_cast<DWORD>(std::size(path)));
      if (!size || size == std::size(path)) {
        versioning::information out;
        out.reason = "module path unavailable or truncated";
        return out;
      }
      return versioning::read_file(path);
    }
    void discover_v1_private_state() {
      if (!versioning::is_legacy(installed_abi) || v1_private_state_checked) return;
      // The encoding is owned by sl.common, whose version can differ from the
      // interposer in a mixed installation. Resolve it once outside callbacks;
      // a delayed load remains eligible at the next existing discovery poll.
      HMODULE module{};
      if (!GetModuleHandleExW(0, L"sl.common.dll", &module)) return;
      dump_metadata::observe_module(module, "streamline", "1.1.1 resource-state adapter");
      auto info = module_version(module);
      const bool supported = versioning::private_state_v1_1_1(info);
      HMODULE pinned{};
      const bool ready = supported && GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
        reinterpret_cast<LPCWSTR>(module), &pinned);
      v1_private_state_checked = true;
      v1_private_state_ready.store(ready, std::memory_order_release);
      char text[640]{};
      std::snprintf(text, sizeof(text),
        "Sunshine Streamline: V1 private resource-state adapter=%s; sl.common fixed=%u.%u.%u.%u strings='%ls'/'%ls'; %s",
        ready ? "enabled" : "unsupported", HIWORD(info.file_ms), LOWORD(info.file_ms), HIWORD(info.file_ls), LOWORD(info.file_ls),
        info.file_text.data(), info.product_text.data(), supported && !ready ? "module lifetime could not be secured" : info.reason);
      message(text);
      FreeLibrary(module);
    }
    void discover() {
      // GetModuleHandleEx takes a temporary reference atomically, preventing
      // unload while exports/version are inspected. It never loads middleware.
      HMODULE module{};
      if (!GetModuleHandleExW(0, L"sl.interposer.dll", &module)) {
        if (!reported_missing) { message("Sunshine Streamline: waiting for an already-loaded sl.interposer.dll; waiting for supported Streamline observation"); reported_missing = true; }
        return;
      }
      auto info = module_version(module);
      const abi version = versioning::classify(info);
      // An exact identity's header, or every candidate of an inexact 2.x family.
      versioning::number number{};
      auto exact = info;
      const auto contract = version == abi::unsupported ? buffers::contract{} :
        versioning::identity(exact, number) ? buffers::for_version(number) : buffers::for_versions(versioning::family(info));
      dump_metadata::observe_module(module, "streamline", versioning::name(version));
      targets functions{
        reinterpret_cast<void *>(GetProcAddress(module, "slSetConstants")),
        reinterpret_cast<void *>(GetProcAddress(module, "slSetTag")),
        reinterpret_cast<void *>(GetProcAddress(module, "slSetTagForFrame")),
        reinterpret_cast<void *>(GetProcAddress(module, "slEvaluateFeature")),
        reinterpret_cast<void *>(GetProcAddress(module, "slGetNewFrameToken")),
        reinterpret_cast<void *>(GetProcAddress(module, "slGetFeatureFunction"))};
      // 1.0.0/1.0.1 shipped namespace exports; 1.0.2 kept that resource layout
      // but introduced the C names. These are the official Windows x64 names.
      if (version == abi::legacy_no_state) {
        if (!functions.constants) functions.constants = reinterpret_cast<void *>(GetProcAddress(module, "?setConstants@sl@@YA_NAEBUConstants@1@II@Z"));
        if (!functions.tag) functions.tag = reinterpret_cast<void *>(GetProcAddress(module, "?setTag@sl@@YA_NPEBUResource@1@W4BufferType@1@IPEBUExtent@1@@Z"));
        if (!functions.evaluate) functions.evaluate = reinterpret_cast<void *>(GetProcAddress(module, "?evaluateFeature@sl@@YA_NPEAXW4Feature@1@II@Z"));
      }
      char identity[768]{};
      std::snprintf(identity, sizeof(identity),
        "Sunshine Streamline: version fixed-file=%u.%u.%u.%u fixed-product=%u.%u.%u.%u strings='%ls'/'%ls' result=%s; adapter=%s; exports constants=%d tag=%d frame-tag=%d evaluate=%d new-frame-token=%d feature-function=%d",
        HIWORD(info.file_ms), LOWORD(info.file_ms), HIWORD(info.file_ls), LOWORD(info.file_ls),
        HIWORD(info.product_ms), LOWORD(info.product_ms), HIWORD(info.product_ls), LOWORD(info.product_ls),
        info.file_text.data(), info.product_text.data(), info.reason, versioning::name(version),
        functions.constants != nullptr, functions.tag != nullptr, functions.tag_for_frame != nullptr, functions.evaluate != nullptr,
        functions.new_frame_token != nullptr, functions.get_feature_function != nullptr);
      message(identity);
      if (version == abi::unsupported || !functions.constants || !functions.tag || !functions.evaluate) {
        message(version == abi::unsupported ?
          "Sunshine Streamline: camera hooks disabled because the version identity or ABI major above is unsupported" :
          "Sunshine Streamline: camera hooks disabled because a required export above is missing");
        permanently_rejected = true;
        FreeLibrary(module);
        return;
      }
      HMODULE pinned_addon{}, pinned_interposer{}, own_module{};
      // Never pin the fixture process executable or another caller's module.
      const bool owned = addon_module && addon_module != GetModuleHandleW(nullptr) &&
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
          reinterpret_cast<LPCWSTR>(&discover), &own_module) && own_module == addon_module;
      if (!owned || !GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
            reinterpret_cast<LPCWSTR>(&discover), &pinned_addon) ||
          !GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
            reinterpret_cast<LPCWSTR>(functions.constants), &pinned_interposer)) {
        message("Sunshine Streamline: module lifetime could not be secured; camera hooks disabled");
        permanently_rejected = true;
        FreeLibrary(module);
        return;
      }
      const bool installed = install_hooks(version, functions, contract);
      FreeLibrary(module);
      if (!installed) {
        observing.store(false, std::memory_order_release);
        permanently_rejected = true;
        message("Sunshine Streamline: inline-hook setup failed; retained code stays pass-through and camera observation is disabled");
        return;
      }
      observing.store(middleware_requested(), std::memory_order_release);
      message(versioning::is_legacy(version) ?
        "Sunshine Streamline: observing 1.x constants, global depth tags and feature evaluation; direct depth capture enabled when supported" :
        "Sunshine Streamline: observing 2.x constants, global/local depth tags and feature evaluation; direct depth capture enabled when supported");
      char candidates_text[128]{};
      for (unsigned i = 0, at = 0; i != contract.candidate_count && at < sizeof(candidates_text); ++i) {
        const auto &v = contract.candidates[i];
        at += unsigned(std::snprintf(candidates_text + at, sizeof(candidates_text) - at, "%s%u.%u.%u.%u", i ? "," : " candidates=",
          v.major, v.minor, v.patch, v.build));
      }
      char buffers_text[448]{};
      std::snprintf(buffers_text, sizeof(buffers_text),
        "Sunshine Streamline: buffer contract %u.%u.%u%s%s last_type=%u ui_alpha_tag=%d ui_color_alpha_tag=%s surveyed=%u ui_alpha_auto=%u",
        contract.version.major, contract.version.minor, contract.version.patch,
        contract.candidate_count > 1 ? " (v2 family, inexact version; a tag is known only where every candidate agrees)" : "",
        candidates_text, contract.last_type,
        contract.raw_ui_alpha == buffers::unknown ? -1 : int(contract.raw_ui_alpha),
        contract.ui_color_and_alpha ? "23" : "none", contract.surveyed ? 1u : 0u,
        buffers::ui_alpha_authenticated(contract) ? 1u : 0u);
      message(buffers_text);
      if (version == abi::v2 && !functions.tag_for_frame)
        message("Sunshine Streamline: optional slSetTagForFrame unavailable; global/evaluation-local tags remain enabled");
      if (version == abi::v2 && !functions.new_frame_token)
        message("Sunshine Streamline: slGetNewFrameToken unavailable; v2 evaluation frame evidence stays untracked");
    }
  }

  // MinHook snapshots every thread on the system (~50 ms with ~10k threads)
  // for each enable or apply. Streamline hook discovery and newly registered
  // FG targets are therefore installed on a pool thread that holds
  // poll_lock; a present-path poll meanwhile skips, as on contention. The
  // add-on is pinned, so a pending callback never outlives its code.
  namespace {
    std::atomic<bool> install_scheduled{false};
    bool fg_targets_pending() {
      if (!TryAcquireSRWLockShared(&fg_lock)) return false;
      bool pending = false;
      for (unsigned i = 0; i != fg_target_count && !pending; ++i)
        pending = fg_targets[i].state.load(std::memory_order_acquire) == 0;
      ReleaseSRWLockShared(&fg_lock);
      return pending;
    }
    // Caller holds poll_lock.
    void install_hook_work() {
      if (!hooks_installed && !permanently_rejected) discover();
      if (middleware_requested() && observing.load(std::memory_order_acquire)) {
        install_fg_hooks();
      }
    }
    void CALLBACK install_on_pool(PTP_CALLBACK_INSTANCE, void *) {
      AcquireSRWLockExclusive(&poll_lock);
      if (!sunshine_addon_lifetime::stopping()) install_hook_work();
      install_scheduled.store(false, std::memory_order_release);
      ReleaseSRWLockExclusive(&poll_lock);
    }
    // Caller holds poll_lock.
    void schedule_install() {
      if (install_scheduled.exchange(true, std::memory_order_acq_rel)) return;
      if (TrySubmitThreadpoolCallback(install_on_pool, nullptr, nullptr)) return;
      install_hook_work(); // Without a pool thread, install here as before.
      install_scheduled.store(false, std::memory_order_release);
    }
  }

  void initialize(HMODULE addon) {
    const bool source_enabled = sunshine_game3d::addon_config::streamline_depth_source();
    const bool call_trace_enabled = sunshine_game3d::addon_config::upscaler_call_trace();
    const bool ngx_source_enabled = sunshine_game3d::addon_config::ngx_depth_source();
    const bool queue_fence_order = sunshine_game3d::addon_config::queue_fence_order();
    native_observer::set_queue_order(queue_fence_order);
    sunshine_upscaler_trace::initialize(addon, call_trace_enabled, ngx_source_enabled);
    // A pool install may still be discovering after device recreation; wait
    // for it before resetting the state it reads and writes.
    AcquireSRWLockExclusive(&poll_lock);
    source_requested.store(false, std::memory_order_release);
    observing.store(false, std::memory_order_release);
    observation_generation.fetch_add(1, std::memory_order_acq_rel);
    addon_module = addon;
    // ReShade may re-register this process-pinned add-on after device recreation.
    // Keep the existing hooks and only start a fresh observation epoch.
    while (!token_metadata.clear()) SwitchToThread();
    release_camera_records();
    AcquireSRWLockExclusive(&presentations_lock);
    lose(loss_diagnostics::reason::lifecycle, __func__, 3382, {});
    ++history_loss; // Camera records were released: every viewport's history restarts.
    ++epoch;
    ReleaseSRWLockExclusive(&presentations_lock);
    next_discovery = 0;
    source_requested.store(source_enabled, std::memory_order_release);
    if (hooks_installed) observing.store(source_enabled, std::memory_order_release);
    ReleaseSRWLockExclusive(&poll_lock);
    message("Sunshine Streamline: diagnostic probe disabled; command/content tracking is off");
  }
  bool source_enabled() { return source_requested.load(std::memory_order_acquire); }
  std::uint64_t depth_observation_revision() { return loss_revision.load(std::memory_order_acquire); }
  bool query_depth_observation_loss(std::uint64_t revision, loss_diagnostics::event &out) {
    return loss_journal.query(revision, out);
  }
  void shutdown() {
    sunshine_upscaler_trace::shutdown();
    source_requested.store(false, std::memory_order_release);
    observing.store(false, std::memory_order_release);
    observation_generation.fetch_add(1, std::memory_order_acq_rel);
    // The shared capture owner retains GPU completion/retirement leases.
    sunshine_game3d::ui_mask::invalidate_all();
    release_camera_records();
  }

  bool present_deferred(std::uint64_t queue, std::uint64_t immediate) {
    if (!deferral_pending.load(std::memory_order_acquire) || !queue || !immediate) return false;
    namespace mask = sunshine_game3d::ui_mask;
    const DWORD incoming = GetLastError();
    const auto ticket = observation_ticket();
    const auto executions = depth_capture::game_executions();
    const auto queue_executions = depth_capture::game_executions_on(queue);
    const auto now = GetTickCount64();
    // This Present's ordinal: it is counted after this call (on_present).
    const auto ordinal = depth_capture::next_present_ordinal();
    const auto source_epoch = epoch.load(std::memory_order_acquire);
    std::array<deferred_frame, slot_count> due;
    unsigned waited = 0;
    unsigned due_count = 0;
    struct ended_frame { std::uint64_t epoch{}, serial{}; std::uint32_t viewport{}; unsigned kinds{}; };
    std::array<ended_frame, slot_count> ended;
    unsigned ended_count = 0;
    auto pin = read_metadata();
    AcquireSRWLockExclusive(&evaluation_supply_lock);
    for (auto &entry : deferred_frames) {
      if (!entry.used) continue;
      // Its frame's Present is past once the observation changed, the
      // viewport's constants named a later frame, or the source age passed
      // at a Present that cannot be the frame's own (expired: a hitch of the
      // source age between the frame's calls and its Present keeps it; until
      // 10-28 it lapsed there, and that Present was mono).
      bool past = !ticket || entry.observation != ticket || entry.epoch != source_epoch;
      // Without the viewport's metadata, frame generation counts as on.
      bool generating = !pin;
      std::uint32_t generated = 0;
      if (!past && pin) for (const auto &record : pin->records) if (record.used && record.viewport == entry.viewport) {
        past = record.camera_serial > entry.serial && !same_frame(record.camera_frame, entry.frame);
        generating = record.frame_generation.known && record.frame_generation.enabled;
        generated = generating ? record.frame_generation.generated_frames : 0;
        break;
      }
      past = past || fg_evaluation.expired(entry.epoch, entry.viewport, entry.evaluations, entry.tick, generated, ordinal, now);
      // Its frame's Present. Without frame generation the game's Present
      // reaches this one directly, so the first after a game list executed
      // following the tag (one with none since is an earlier frame's). With
      // it, earlier frames' Presents (paced or generated) come while the game
      // submits this frame's work on any queue: the first after DLSS-G
      // evaluated a frame following the tag, while DLSS-G evaluates every
      // frame in this scope; else the first after a game list executed on
      // this presenting queue since the tag (deferral_evaluations).
      const bool reached = !generating ? entry.executions != executions :
        fg_evaluation.presented(entry.epoch, entry.viewport, entry.evaluations, queue_executions > entry.executions, now, ordinal);
      if (!past && !reached) {
        waited += generating;
        continue;
      }
      if (past) ended[ended_count++] = {entry.epoch, entry.serial, entry.viewport, deferred_kinds(entry)};
      else due[due_count++] = std::move(entry);
      entry = {};
    }
    count_deferred();
    ReleaseSRWLockExclusive(&evaluation_supply_lock);
    pin.reset();
    if (waited) deferrals.waited.fetch_add(waited, std::memory_order_relaxed);
    for (unsigned n = 0; n != ended_count; ++n) {
      deferrals.lapsed.fetch_add(kind_count(ended[n].kinds), std::memory_order_relaxed);
      hand_over(ended[n].epoch, ended[n].viewport, ended[n].kinds, ended[n].serial);
    }
    bool recorded = false;
    // ReShade's own list records these copies; its list events never report it.
    if (due_count) depth_capture::observe_runtime_list(immediate, queue);
    for (unsigned n = 0; n != due_count; ++n) {
      auto &entry = due[n];
      const auto sequence = ++call_sequence;
      if (entry.depth.count) {
        // A refusal changes nothing: the earlier capture holds (S6), and the
        // FG observation stays as it was (pixels_required).
        depth_capture::record_diagnostic diagnostic;
        bool copied = false;
        // FG Off retired the FG role of the viewport's lineage after this
        // frame's tag call (a late Off) or keeps it off: the copy is the
        // frame's own depth of the lineage, its ordinary source (source 0,
        // superseding the FG view), and no FG scope's (the 10-28 review's
        // P1; until then it became the FG view again beside the SR view).
        const bool retired = entry.depth.candidates[0].frame_generation_input &&
          depth_capture::frame_generation_retired(entry.epoch, entry.viewport, entry.serial);
        const auto fg_key = sunshine_scene_depth::encode_source_id(sunshine_scene_depth::fg_depth_key(entry.viewport));
        for (std::size_t i = 0; i != entry.depth.count && !copied; ++i) {
          auto value = entry.depth.candidates[i];
          value.sequence = sequence; value.present_queue = queue;
          value.own_frame = true; value.at_tag_call = false; value.pixels_required = true;
          if (retired) { value.frame_generation_input = false; value.source_id = 0; }
          const auto captured = depth_capture::nominate(immediate, value, &diagnostic);
          if (captured && diagnostic.result == depth_capture::status::recorded) {
            depth_capture::begin_evaluation(value.epoch, sequence, value.viewport, value.provider, value.source_id,
              retired ? fg_key : entry.depth.superseded);
            depth_capture::finish(captured, true);
            copied = true;
          } else if (captured) depth_capture::finish(captured, false);
        }
        if (!retired) {
          remember_tag_copy(entry.epoch, entry.viewport, copied, entry.serial, entry.frame);
          if (!copied) miss_frame_generation(entry.epoch, entry.viewport, entry.serial);
        }
        if (copied) {
          recorded = true;
          deferrals.depth_copied.fetch_add(1, std::memory_order_relaxed);
        } else {
          deferrals.depth_refused.fetch_add(1, std::memory_order_relaxed);
          deferrals.refusal.store(unsigned(diagnostic.stage), std::memory_order_relaxed);
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
          evaluation_snapshot refused;
          refused.epoch = entry.epoch; refused.sequence = sequence; refused.viewport = entry.viewport;
          refused.frame_generation_input = true;
          remember_refusal(refused, sunshine_depth_refusal::gate::capture, nullptr, &diagnostic);
#endif
        }
      }
      // One new observation sequence per tag call, in call order, so a call's
      // UI kinds stay one batch (exact HUD-less pairs) and no two calls merge.
      std::array<std::uint64_t, 4> calls{}, renumbered{};
      unsigned call_count = 0;
      for (const auto &item : entry.ui) if (item.used) {
        unsigned at = 0;
        while (at != call_count && calls[at] < item.input.sequence) ++at;
        if (at != call_count && calls[at] == item.input.sequence) continue;
        for (unsigned i = call_count; i != at; --i) calls[i] = calls[i - 1];
        calls[at] = item.input.sequence; ++call_count;
      }
      for (unsigned i = 0; i != call_count; ++i) renumbered[i] = i ? ++call_sequence : sequence;
      for (unsigned slot = 0; slot != entry.ui.size(); ++slot) {
        auto &item = entry.ui[slot];
        if (!item.used) continue;
        unsigned call = 0;
        while (calls[call] != item.input.sequence) ++call;
        item.input.sequence = renumbered[call];
        item.input.present_queue = queue; item.input.own_frame = true;
        item.where.source = item.input;
        item.where.command = immediate;
        const auto attempt = mask::begin(item.where, item.input);
        mask::finish(attempt, true);
        remember_ui_tag_copy(entry.epoch, entry.viewport, slot, static_cast<bool>(attempt.ticket), true);
        if (attempt.ticket) {
          recorded = true;
          deferrals.ui_copied.fetch_add(1, std::memory_order_relaxed);
        } else deferrals.ui_refused.fetch_add(1, std::memory_order_relaxed);
      }
    }
    report_deferrals();
    SetLastError(incoming);
    return recorded;
  }

  frame_generation::attempt observe_frame_generation_inputs(const frame_generation::inputs &value) {
    namespace mask = sunshine_game3d::ui_mask;
    using frame_generation::depth_outcome;
    using frame_generation::parameter;
    using frame_generation::ui_outcome;
    const DWORD incoming = GetLastError();
    frame_generation::attempt out;
    // An absent key changes nothing and is reported as absent in any scope.
    out.depth = value.depth.state == parameter::absent ? depth_outcome::absent :
      value.depth.state == parameter::null ? depth_outcome::null_resource : depth_outcome::not_offered;
    for (unsigned i = 0; i != value.ui.size(); ++i)
      out.ui_result[i] = value.ui[i].state == parameter::absent ? ui_outcome::absent : ui_outcome::no_fg_scope;
    // The scope: the one viewport whose FG options were observed enabled in
    // this epoch. Without it no input belongs to any Streamline FG scope.
    const auto ticket = observation_ticket();
    frame_generation_snapshot fg;
    if (!ticket || !query_frame_generation(UINT32_MAX, fg) || !fg.enabled) {
      if (out.depth == depth_outcome::not_offered) out.depth = depth_outcome::no_fg_scope;
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
      remember_frame_generation_refusal(out, 0, 0, UINT32_MAX);
#endif
      SetLastError(incoming);
      return out;
    }
    const auto loss = loss_revision.load(std::memory_order_acquire);
    const auto source_epoch = epoch.load(std::memory_order_acquire);
    if (fg.epoch != source_epoch || !current(ticket)) {
      if (out.depth == depth_outcome::not_offered) out.depth = depth_outcome::scope_changed;
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
      remember_frame_generation_refusal(out, fg.epoch, 0, fg.viewport);
#endif
      SetLastError(incoming);
      return out;
    }
    out.observation = ticket; out.loss = loss;
    // One observation sequence for every input of this evaluation, ordered
    // with every Streamline call (the capture owner's FG view and the UI
    // request's newest boundary compare it with tag sequences).
    const auto serial = ++call_sequence;
    const auto tick = GetTickCount64();
    // This frame's Present comes next (present_deferred).
    AcquireSRWLockExclusive(&evaluation_supply_lock);
    fg_evaluation.evaluated(fg.epoch, fg.viewport, tick, depth_capture::next_present_ordinal(), fg.generated_frames);
    ReleaseSRWLockExclusive(&evaluation_supply_lock);
    const auto source_id = sunshine_scene_depth::encode_source_id(sunshine_scene_depth::fg_depth_key(fg.viewport));
    if (out.depth == depth_outcome::not_offered) {
      if (!value.depth.area_valid) out.depth = depth_outcome::subrect;
      else if (!source_requested.load(std::memory_order_acquire) || !depth_capture::active()) out.depth = depth_outcome::inactive;
      // The frame's own FG tag copied it first, or its copy waits for the
      // frame's Present: no second copy (a later frame without an FG tag of
      // its own is DLSS-G's).
      else if (tags_supply_frame(fg.epoch, fg.viewport, evaluated_frame(fg.viewport))) out.depth = depth_outcome::tag_copied;
      // The viewport's own camera evidence, exactly as for an unframed FG tag
      // (normalize_depth_frame): its history revision, its recent camera's
      // depth direction and its steady depth coefficients. No frame
      // correlation, jitter or reset pulse is claimed. Busy or lost metadata
      // refuses, as a tag boundary does.
      else if (const auto viewport = capture_evaluation(fg.viewport, 1000, frame_identity{},
                 reinterpret_cast<void *>(value.command), ticket, serial, loss, nullptr, nullptr, false);
               viewport.status == evidence_status::busy || viewport.status == evidence_status::inactive ||
               viewport.epoch != fg.epoch)
        out.depth = depth_outcome::scope_changed;
      else {
        depth_capture::observe_command(value.command);
        depth_capture::input input;
        input.provider = sunshine_scene_depth::provider_kind::streamline;
        input.epoch = fg.epoch; input.sequence = serial; input.tick = tick; input.viewport = fg.viewport;
        input.observation_revision = loss; input.feedback = viewport.feedback;
        input.frame_generation_input = true; input.source_id = source_id;
        // Device depth (DLSS-G linearizes 1/(1-d), or 1/d when inverted).
        // DLSSG.DepthInverted names its direction; without it the viewport
        // camera's does. The steady coefficients apply only when both agree
        // (or only the camera names it): DLSSG.CameraNear/Far are not used.
        input.projection.encoding = sunshine_scene_depth::depth_encoding::device;
        const bool camera_inverted = viewport.viewport_inverted != 0;
        const bool agree = !value.inverted_known || !viewport.viewport_direction || value.inverted == camera_inverted;
        input.projection.direction_supplied = value.inverted_known || viewport.viewport_direction;
        input.projection.reversed = value.inverted_known ? value.inverted : viewport.viewport_direction && camera_inverted;
        if (agree && viewport.viewport_direction && viewport.viewport_projection_ready) {
          input.projection.depth_offset = viewport.viewport_projection.depth_offset;
          input.projection.depth_scale = viewport.viewport_projection.depth_scale;
          input.projection.supplied = true;
        }
        input.resource.native = value.depth.native;
        input.resource.area = value.depth.area;
        input.resource.kind = sunshine_scene_depth::resource_kind::raw_depth;
        // No API states this resource's state at the call: only a state
        // observed on the evaluation's own recording is acceptable.
        sunshine_api_contract::apply(sunshine_api_contract::ngx::frame_generation_input, input);
        input.at_tag_call = true; // Copied inside the call that supplied it.
        input.pixels_required = true;
        depth_capture::record_diagnostic diagnostic;
        input.source = depth_capture::retain_source(input.resource.native, &diagnostic);
        if (!input.source) {
          out.depth = depth_outcome::capture; out.depth_stage = diagnostic.stage;
        } else {
          input.source_present_generation = depth_capture::source_present_generation(input.source);
          // A live scope's attempt replaces its FG observation first, exactly
          // as an FG tag boundary does, so a refusal withdraws the observation,
          // never finished pixels: it is a missing frame, for which the scope's
          // newest unshown copy stands in (depth_capture::select_capture). A
          // silent scope's refusal changes nothing: only a recorded copy
          // becomes the viewport's FG observation (replacing its ordinary one).
          // Either way a refused frame has no FG copy, so the scope owns no
          // depth for it (frame_generation_missed).
          // A live scope's attempt is a nomination in progress from its begin.
          // An evaluation that began before FG Off retired the viewport's FG
          // role (DLSS-G's thread racing the game's options call) or while it
          // stays retired is the frame's own depth of the display lineage,
          // its ordinary source, superseding the FG view (the 10-28 review's
          // P1; until then it became the FG view again beside the SR view).
          const bool retired = depth_capture::frame_generation_retired(fg.epoch, fg.viewport, serial);
          if (retired) { input.frame_generation_input = false; input.source_id = 0; }
          const bool live = !retired && depth_capture::frame_generation_scope_live(fg.epoch, fg.viewport, fg.enabled_tick);
          depth_capture::evaluation_nomination nominating;
          if (live) nominating.begin(fg.epoch, serial, fg.viewport, sunshine_scene_depth::provider_kind::streamline, source_id, 0);
          const auto captured = nominating.nominate(value.command, input, &diagnostic);
          nominating.end();
          if (captured && diagnostic.result == depth_capture::status::recorded) {
            if (!live) depth_capture::begin_evaluation(fg.epoch, serial, fg.viewport,
              sunshine_scene_depth::provider_kind::streamline, input.source_id, retired ? source_id : 0);
            if (!retired) remember_evaluation_supply(fg.epoch, fg.viewport, tick);
            out.depth_ticket = captured; out.depth = depth_outcome::copied;
          } else {
            if (captured) depth_capture::finish(captured, false);
            if (!retired) depth_capture::frame_generation_missed(fg.epoch, fg.viewport, serial);
            out.depth = depth_outcome::capture; out.depth_stage = diagnostic.stage;
          }
        }
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
        remember_frame_generation_refusal(out, fg.epoch, serial, fg.viewport, diagnostic.result);
#endif
      }
    }
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
    if (out.depth != depth_outcome::capture) remember_frame_generation_refusal(out, fg.epoch, serial, fg.viewport);
#endif
    if (std::any_of(value.ui.begin(), value.ui.end(), [](const auto &item) { return item.state != parameter::absent; })) {
      const bool interested = mask::interested(source_epoch, loss, fg.viewport, tick);
      for (unsigned i = 0; i != value.ui.size(); ++i) {
        const auto &item = value.ui[i];
        auto &result = out.ui_result[i];
        if (item.state == parameter::absent) continue;
        if (!interested) { result = ui_outcome::no_request; continue; }
        if (!item.area_valid) { result = ui_outcome::subrect; continue; }
        // The game's own tag of this kind copied it first, or its copy waits
        // for this frame's Present: no second copy, and its exact pairs stay
        // the tags' own (ui_tags_supply).
        if (ui_tags_supply(source_epoch, fg.viewport, i)) {
          result = ui_outcome::tag_copied;
          continue;
        }
        depth_capture::input input;
        input.epoch = source_epoch; input.sequence = serial; input.tick = tick;
        input.viewport = fg.viewport; input.observation_revision = loss;
        input.source_id = source_id; input.frame_generation_input = true;
        input.resource.native = item.state == parameter::present ? item.native : 0;
        input.resource.area = item.area;
        sunshine_api_contract::apply(sunshine_api_contract::ngx::frame_generation_input, input);
        input.force_snapshot = true; input.at_tag_call = true;
        depth_capture::record_diagnostic retained;
        input.source = depth_capture::retain_source(input.resource.native, &retained);
        input.source_present_generation = depth_capture::source_present_generation(input.source);
        if (input.source) { input.resource.width = retained.width; input.resource.height = retained.height; }
        mask::boundary where;
        where.source = input; where.command = value.command; where.tag_scope = 2;
        where.kind = mask::capture_order[i];
        where.source_present_generation = input.source_present_generation;
        where.format = input.source ? retained.format : 0;
        where.frame_generation_evaluation = true;
        out.ui[i] = mask::begin(where, input);
        result = item.state == parameter::present ? ui_outcome::offered : ui_outcome::null_resource;
      }
    }
    SetLastError(incoming);
    return out;
  }
  void finish_frame_generation_inputs(const frame_generation::attempt &value, bool successful) {
    if (!value.observation) return;
    const DWORD incoming = GetLastError();
    // A copy survives a successful SDK call; the UI inputs also need the
    // observation they were admitted under, as Streamline tags do.
    const bool still_current = current(value.observation);
    if (value.depth_ticket)
      depth_capture::finish(value.depth_ticket, successful && still_current, depth_capture::capture_failure::evaluation_failed);
    const bool ui_current = successful && still_current && value.loss == loss_revision.load(std::memory_order_acquire);
    for (const auto &attempt : value.ui) sunshine_game3d::ui_mask::finish(attempt, ui_current);
    SetLastError(incoming);
  }

  const char *name(evidence_status value) {
    switch (value) {
      case evidence_status::inactive: return "inactive";
      case evidence_status::busy: return "snapshot-busy";
      case evidence_status::missing_evaluation: return "missing-evaluation";
      case evidence_status::evaluation_failed: return "evaluation-failed";
      case evidence_status::observation_lost: return "observation-lost";
      case evidence_status::untracked_frame: return "untracked-frame";
      case evidence_status::missing_constants: return "missing-frame-constants";
      case evidence_status::rejected_constants: return "rejected-constants";
      case evidence_status::invalid_camera: return "invalid-camera";
      case evidence_status::camera_reset: return "camera-reset";
      case evidence_status::missing_depth: return "missing-depth";
      case evidence_status::unsupported_depth: return "unsupported-depth-tag";
      case evidence_status::depth_not_ready: return "selected-depth-not-ready";
      case evidence_status::source_mismatch: return "selected-source-mismatch";
      case evidence_status::extent_mismatch: return "selected-extent-mismatch";
      case evidence_status::stale: return "stale-evaluation";
      case evidence_status::ambiguous_viewport: return "ambiguous-viewport";
      case evidence_status::source_associated_evaluation: return "source-associated-evaluation-reshade-frame-unverified";
      case evidence_status::command_associated_evaluation: return "command-associated-evaluation-content-unverified";
      case evidence_status::tracked_content_evaluation: return "tracked-content-evaluation-coverage-incomplete";
    }
    return "unknown";
  }

  namespace {
    // Diagnostic only: whether this game also sends Streamline camera constants,
    // reported beside NGX capture. Never a cross-API frame match.
    void report_camera_availability(std::uint64_t now) {
      if (!sunshine_upscaler_trace::capture_enabled() || !hooks_installed) return;
      static std::uint64_t camera_report_at{}, camera_report_count{UINT64_MAX};
      auto camera_pin = now >= camera_report_at ? read_metadata() : metadata_owner::read_pin{};
      if (camera_pin) {
        unsigned cameras = 0, valid_recent = 0;
        camera_data camera_sample;
        camera_key camera_sample_key;
        camera_validation camera_sample_validation;
        std::uint64_t camera_sample_tick{};
        struct depth_tag_sample {
          depth_tag value;
          std::uint64_t tick{};
          origin scope{};
          bool present{}, supported{};
        };
        std::array<depth_tag_sample, 3> depth_samples{};
        for (const auto &record : camera_pin->records) if (record.used && record.has_camera) {
          ++cameras;
          const auto checked = validate(record.camera);
          valid_recent += sunshine_freshness::fresh_le(record.camera_tick, now, sunshine_freshness::camera_age_ms) &&
            record.decoded == decode_status::ok && checked.valid();
          if (record.decoded == decode_status::ok && record.camera_tick >= camera_sample_tick) {
            camera_sample = record.camera;
            camera_sample_key = record.key;
            camera_sample_validation = checked;
            camera_sample_tick = record.camera_tick;
            depth_samples = {};
            for (const auto *group : {&record.active_tags, &record.global_tags, &record.local_tags}) {
              for (const auto &tag : *group) {
                if (!tag.present || !buffers::is_depth(tag.value.type)) continue;
                auto &sample = depth_samples[slot_of(tag.value.type)];
                if (!sample.present || tag.tick > sample.tick)
                  sample = {tag.value, tag.tick, tag.source, tag.present, tag.supported};
              }
            }
          }
        }
        const auto calls = constant_calls.load(std::memory_order_relaxed);
        camera_pin.reset();
        camera_report_at = now + 5000;
        if (calls != camera_report_count) {
          // vram_warnings: SDK calls answered with eWarnOutOfVRAM since the
          // observation began, each accepted as a success.
          char availability[352]{};
          std::snprintf(availability, sizeof(availability),
            "Sunshine Streamline camera availability: constants_calls=%llu cameras=%u recent_valid_projection=%u vram_warnings=%llu; NGX_frame_match=unproven; availability does not authorize cross-API scale",
            static_cast<unsigned long long>(calls), cameras, valid_recent,
            static_cast<unsigned long long>(vram_warnings.load(std::memory_order_relaxed)));
          message(availability);
          if (camera_sample_tick) {
            char detail[1800]{};
            std::snprintf(detail, sizeof(detail),
              "Sunshine Streamline camera sample: viewport=%u frame=%llu explicit_frame=%u age_ms=%llu projection=%s A=%.9g B=%.9g matrix_near=%.9g matrix_far=%.9g infinite_far=%u checked_fov_rad=%.9g checked_aspect=%.9g declared_near=%.9g declared_far=%.9g declared_fov_rad=%.9g declared_aspect=%.9g inverted=%u orthographic=%u reset=%u; checked fields may be incomplete on failure; observation only, NGX_frame_match=unproven",
              camera_sample_key.viewport, static_cast<unsigned long long>(camera_sample_key.frame_key),
              camera_sample_key.explicit_frame ? 1u : 0u,
              static_cast<unsigned long long>(camera_sample_tick <= now ? now - camera_sample_tick : 0),
              name(camera_sample_validation.status), camera_sample_validation.depth_offset,
              camera_sample_validation.depth_scale, camera_sample_validation.near_from_matrix,
              camera_sample_validation.far_from_matrix, camera_sample_validation.infinite_far ? 1u : 0u,
              camera_sample_validation.fov_from_matrix, camera_sample_validation.aspect_from_matrix,
              camera_sample.near_plane, camera_sample.far_plane, camera_sample.fov, camera_sample.aspect,
              unsigned(camera_sample.depth_inverted), unsigned(camera_sample.orthographic),
              unsigned(camera_sample.reset));
            message(detail);
            char tag_details[3][480]{};
            for (unsigned i = 0; i < depth_samples.size(); ++i) {
              const auto &sample = depth_samples[i];
              const auto &tag = sample.value;
              std::snprintf(tag_details[i], sizeof(tag_details[i]),
                "type=%u present=%u supported=%u scope=%u native=0x%llx declared_size=%ux%u area_xywh=%u,%u,%u,%u lifetime=%u age_ms=%llu tag_viewport=%u frame=%llu explicit_frame=%u",
                observed_tag_types[i], sample.present ? 1u : 0u,
                sample.supported ? 1u : 0u, static_cast<unsigned>(sample.scope),
                static_cast<unsigned long long>(tag.native_resource), tag.resource_width, tag.resource_height,
                tag.area.left, tag.area.top, tag.area.width, tag.area.height, tag.lifecycle,
                static_cast<unsigned long long>(sample.present && sample.tick <= now ? now - sample.tick : UINT64_MAX),
                tag.viewport, static_cast<unsigned long long>(tag.frame_key), tag.explicit_frame ? 1u : 0u);
            }
            std::snprintf(detail, sizeof(detail),
              "Sunshine Streamline depth tag sample: camera_viewport=%u depth={%s} highres_depth={%s} linear_depth={%s}; observation only, NGX_resource_overlap=unproven",
              camera_sample_key.viewport, tag_details[0], tag_details[1], tag_details[2]);
            message(detail);
          }
          camera_report_count = calls;
        }
      }
    }
  }

  poll_guard::~poll_guard() {
    if (acquired_) ReleaseSRWLockExclusive(&poll_lock);
  }

  poll_guard try_poll() {
    return poll_guard(TryAcquireSRWLockExclusive(&poll_lock) != FALSE);
  }

  bool middleware_requested_for_poll() {
    return middleware_requested();
  }

  void discovery_step(bool api_requested, std::uint64_t now) {
    if (api_requested && now >= next_discovery) {
      next_discovery = now + 1000;
      if (!hooks_installed && !permanently_rejected) schedule_install();
      if (hooks_installed) discover_v1_private_state();
      if (hooks_installed) discover_fg_options();
    }
  }

  void coverage_step() {
    // Call routing is independent of per-draw diagnostics and source selection.
    // A zero count is only useful together with actual hook coverage.
    sunshine_upscaler_trace::set_streamline_coverage(hooks_installed);
  }

  void camera_availability_step(std::uint64_t now) {
    report_camera_availability(now);
  }

  void depth_capture_step() {
    if (!minhook_initialized) {
      const auto initialized = MH_Initialize();
      minhook_initialized = initialized == MH_OK || initialized == MH_ERROR_ALREADY_INITIALIZED;
    }
    if (minhook_initialized) depth_capture::poll();
  }

  void queue_order_step(std::uint64_t now) {
    char order_line[1024];
    if (depth_capture::queue_order_line(now, order_line, sizeof(order_line)))
      sunshine_log::message(reshade::log::level::info, order_line);
  }

  void pending_installs_step() {
    if (middleware_requested() && observing.load(std::memory_order_acquire) && fg_targets_pending())
      schedule_install();
  }

#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
  // The test add-on's exports (test_streamline_camera_probe_exports.cpp) read
  // camera data through this instantiation.
  SUNSHINE_PROBE_INTERNAL {
    template bool read(const camera_data *, camera_data &);
  }
#endif

#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
  namespace testing {
    void arm_nomination_tap(bool armed) {
      nomination_tap_armed = armed;
      nomination_tap_pending = false;
    }
    bool latest_nomination(std::uint32_t viewport, nomination &out) {
      for (const auto &entry : nomination_tap) if (nomination_tap_current(entry) && entry.value.snapshot.viewport == viewport) {
        out = entry.value; return true;
      }
      return false;
    }
    thread_local metadata_owner::transaction held_metadata_writer;
    thread_local metadata_owner::read_pin held_metadata_reader;
    void lock_records() { while (!(held_metadata_writer = metadata.try_write())) SwitchToThread(); }
    void unlock_records() { held_metadata_writer.reset(); }
    void lock_records_shared() { held_metadata_reader = read_metadata(); }
    void unlock_records_shared() { held_metadata_reader.reset(); }
    thread_local token_owner::transaction held_token_writer;
    thread_local std::array<token_owner::read_pin, 8> held_token_readers;
    void lock_tokens() { while (!(held_token_writer = token_metadata.try_write())) SwitchToThread(); }
    void unlock_tokens() { held_token_writer.reset(); }
    bool pin_tokens(unsigned index) {
      if (index >= held_token_readers.size()) return false;
      held_token_readers[index] = token_metadata.read();
      return bool(held_token_readers[index]);
    }
    void unpin_tokens() { for (auto &pin : held_token_readers) pin.reset(); }
    void age_frame_generation(std::uint64_t ms) {
      auto update = metadata.try_write();
      while (!update) { SwitchToThread(); update = metadata.try_write(); }
      for (auto &record : update->records) {
        auto &tick = record.frame_generation.enabled_tick;
        if (tick) tick = tick > ms ? tick - ms : 1;
      }
      update.commit();
    }
    void arm_entry_policy() { entry_policy_armed = true; entry_policy_observed = false; }
    bool entry_policy(bool &selected, bool &admissible) {
      selected = entry_path_selected; admissible = entry_source_admitted;
      return entry_policy_observed;
    }
    bool discover_frame_generation_hooks() { discover_fg_options(); return install_fg_hooks(); }
    bool install(const versioning::number &version, const testing::targets &functions) {
      if (!middleware_requested() && !sunshine_upscaler_trace::enabled()) return false;
      // discover()'s selection for a version resource naming exactly this version.
      versioning::information resource;
      resource.fixed_valid = true;
      resource.file_ms = resource.product_ms = MAKELONG(version.minor, version.major);
      resource.file_ls = resource.product_ls = MAKELONG(version.build, version.patch);
      const abi native_version = versioning::classify(resource);
      const auto contract = native_version == abi::unsupported ? buffers::contract{} : buffers::for_version(version);
      const bool success = install_hooks(native_version, {functions.constants, functions.tag, functions.tag_for_frame, functions.evaluate, functions.new_frame_token, functions.get_feature_function}, contract);
      if (success) observing.store(middleware_requested(), std::memory_order_release);
      sunshine_upscaler_trace::set_streamline_coverage(success);
      return success;
    }
    counters counts() {
      return {constant_calls.load(), tag_calls.load(), evaluation_calls.load(), dropped.load(), invalid.load(), vram_warnings.load(),
        depth_untags.load()};
    }
    void lose_observation() { dropped_observation(loss_diagnostics::reason::test_injected, __func__, 4488, {}); }
    bool latest(std::uint32_t viewport, std::uint64_t &resource, bool &has_camera, bool &explicit_same_frame, bool &same_token_address) {
      auto pin = read_metadata();
      if (!pin) return false;
      bool found = false;
      for (const auto &record : pin->records) if (record.used && record.viewport == viewport) {
        const tag_record *chosen = nullptr;
        for (const auto &tag : record.global_tags) if (!chosen && tag.present) chosen = &tag;
        for (const auto &tag : record.local_tags) if (tag.present) { chosen = &tag; break; }
        const tag_record empty;
        const auto &tag = chosen ? *chosen : empty;
        resource = tag.value.native_resource;
        has_camera = record.has_camera;
        explicit_same_frame = tag.present && match(record.key, tag.value) == match_status::exact_frame;
        same_token_address = record.camera_token && record.camera_token == tag.token;
        found = tag.present;
      }
      return found;
    }
    void clear() {
      sunshine_game3d::ui_mask::invalidate_all();
      observing.store(false, std::memory_order_release);
      if (minhook_initialized) {
        // No other thread exists in the fixture; production never does this.
        MH_DisableHook(MH_ALL_HOOKS);
        MH_Uninitialize();
      }
      for (unsigned i = 0; i != fg_target_count; ++i) {
        if (fg_targets[i].owner) FreeLibrary(fg_targets[i].owner);
        fg_targets[i].entry = nullptr; fg_targets[i].owner = nullptr;
        fg_targets[i].original = nullptr; fg_targets[i].state = 0;
      }
      fg_target_count = 0; current_fg_target = nullptr; fg_available = false;
      fg_loss_revision = 0;
      original_feature_function_v2 = nullptr;
      hooks_installed = minhook_initialized = false;
      v1_private_state_ready.store(false, std::memory_order_release);
      v1_private_state_checked = false;
      installed_abi = sunshine_streamline::abi::unsupported;
      installed_targets = {};
      constant_calls = tag_calls = evaluation_calls = dropped = invalid = vram_warnings = depth_untags = fg_refusals = 0;
      next_compatibility_report = next_untag_report = next_fg_refusal_report = 0;
      initialize(nullptr);
    }
  }
#endif
}

#include "sl_observation.inl"
#include "sl_nomination.inl"
