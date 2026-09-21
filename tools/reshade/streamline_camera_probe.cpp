// SPDX-License-Identifier: GPL-3.0-only
#include "streamline_camera_probe.h"
#include "addon_lifetime.h"
#include "streamline_camera_data.h"
#include "streamline_camera_version.h"
#include "streamline_v1_resource_state.h"
#include "upscaler_call_trace.h"
#include "game3d_diagnostic_metadata.h"
#include "game3d_ui_mask.h"
#include "observer_snapshot.h"

#include <MinHook.h>
#include <reshade.hpp>
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
  namespace {
    namespace dump_metadata = sunshine_game3d::diagnostic;
    dump_metadata::stamp dump_stamp(std::uint64_t epoch, std::uint64_t sequence, std::uint64_t tick,
        std::uint32_t viewport = UINT32_MAX, std::uint64_t token = 0, std::uint64_t frame = 0,
        bool numeric = false, std::uint64_t command = 0) {
      return {sunshine_game3d::diagnostic_metadata_generation(), epoch, sequence, tick, token, frame, command, viewport, numeric};
    }
    using abi = versioning::abi;
    enum class origin { global, frame, local };
    constexpr unsigned slot_count = 4, tag_limit = 32, chain_limit = 32, frame_count = 32;
    constexpr std::array<std::uint32_t, 7> observed_tag_types{0, 48, 49, 2, 3, 4, 53};
    unsigned tag_index(std::uint32_t type);
    struct tag_record {
      depth_tag value;
      std::uint64_t tick{};
      std::uintptr_t token{};
      origin source{};
      bool present{}, supported{};
      std::uint64_t serial{}, loss{};
      frame_identity frame;
      std::uint64_t resource_lifetime{}, resource_device{};
      std::uint32_t native_state{0xffffffffu};
      bool state_requires_observation{};
      depth_capture::source_ref direct_source;
      std::uint64_t source_present_generation{};
      double precision_scale{1.0}, precision_bias{};
    };
    struct viewport_record {
      camera_data camera;
      camera_key key;
      decode_status decoded{decode_status::short_buffer};
      std::array<tag_record, 7> global_tags{}, local_tags{};
      std::array<tag_record, 7> active_tags{};
      evaluation_snapshot evaluation;
      frame_generation_snapshot frame_generation;
      frame_identity camera_frame;
      std::uintptr_t camera_token{};
      std::uint64_t camera_tick{}, camera_serial{}, changed{}, evaluation_tick{};
      std::uint32_t viewport{}, feature{};
      bool used{}, has_camera{};
    };
    struct targets { void *constants{}, *tag{}, *tag_for_frame{}, *evaluate{}, *new_frame_token{}, *get_feature_function{}; };
    struct batch {
      std::array<tag_record, 7> tags{}; // Three depth and four separately retained color kinds.
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
      std::array<tag_record, 7> tags{};
      std::uint64_t camera_serial{}, camera_tick{}, loss{}, changed{};
      std::uint32_t viewport{};
      bool used{}, has_camera{};
    };
    struct token_record { frame_identity frame; };
    struct presentation_record {
      presentation_snapshot value;
      frame_identity latest_started;
      HANDLE thread{};
      std::uint64_t pending_sequence{};
    };

    // POD/atomic storage remains live in the pinned add-on after AddonUninit. In
    // particular there is no destructible C++ mutex or worker used by detours.
    SRWLOCK presentations_lock = SRWLOCK_INIT;
    SRWLOCK poll_lock = SRWLOCK_INIT;
    SRWLOCK commands_lock = SRWLOCK_INIT;
    SRWLOCK source_resources_lock = SRWLOCK_INIT;
    struct source_resource { std::uint64_t native{}, lifetime{}, device{}; };
    std::array<source_resource, 512> source_resources{};
    std::atomic<bool> source_resources_lost{};
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
    std::atomic<unsigned> source_lifecycle_waiters{};
#endif
    commands::tracker command_tracker;
    content::ledger content_ledger;
    bool content_observing{}; // Guarded by commands_lock.
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
    std::atomic<bool> test_content_requested{};
    std::atomic<bool> test_capture_requested{};
#endif
    std::atomic<bool> command_observation_lost{};
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
    presentation_record presentations[32]{};
    std::atomic<std::uint64_t> presentation_loss{}, presentation_dropped{}, presentation_generation{};
    std::atomic<bool> pcl_available{};
    struct pcl_target {
      void *entry{};
      HMODULE owner{};
      std::atomic<abi_v2::pcl_set_marker> original{};
      std::atomic<unsigned> state{}; // 0 pending, 1 installed, 2 rejected
    };
    std::array<pcl_target, 4> pcl_targets;
    unsigned pcl_target_count{};
    SRWLOCK pcl_lock = SRWLOCK_INIT;
    std::atomic<void *> current_pcl_target{};
    std::atomic<std::uint64_t> pcl_calls{}, pcl_getter_calls{};
    std::atomic<bool> observing{};
    std::atomic<bool> requested{};
    std::atomic<bool> source_requested{};
    std::atomic<std::uint64_t> observation_generation{};
    std::atomic<std::uint64_t> constant_calls{}, tag_calls{}, evaluation_calls{}, dropped{}, invalid{};
    std::atomic<std::uint64_t> call_sequence{}, loss_revision{};
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
    std::uint64_t next_discovery{}, next_report{}, last_report_activity{UINT64_MAX};
    selected_depth last_selected{};
    HMODULE addon_module{};
    abi installed_abi{};
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

    // Pinned v2.7.30 sl_core_types.h/sl_dlss_g.h prefixes. These are copied,
    // never called as SDK objects or followed past the supported version.
    constexpr guid precision_guid{0x98f6e9ba, 0x8d16, 0x4831, {0xa8,0x02,0x4d,0x3b,0x52,0xff,0x26,0xbf}};
    constexpr guid fg_options_guid{0xfac5f1cb, 0x2dfd, 0x4f36, {0xa1,0xe6,0x3a,0x9e,0x86,0x52,0x56,0xc5}};
    struct precision_info { base_structure base; std::uint32_t formula{}; float bias{}, scale{1}; };
    struct fg_options_prefix { base_structure base; std::uint32_t mode{}, generated_frames{}; };
    using fg_set_options = std::int32_t (*)(const abi_v2::viewport &, const fg_options_prefix &);
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

    void release_camera_records() {
      // Lifecycle only: callbacks are already closed. Readers never delay this
      // clear; retired source leases are released outside mutation ownership.
      while (!metadata.clear()) SwitchToThread();
    }

    bool same(const guid &a, const guid &b) { return std::memcmp(&a, &b, sizeof(guid)) == 0; }
    bool read_bytes(const void *source, void *destination, std::size_t size) {
      SIZE_T copied = 0;
      return source && ReadProcessMemory(GetCurrentProcess(), source, destination, size, &copied) && copied == size;
    }
    template<class T> bool read(const T *source, T &destination) { return read_bytes(source, &destination, sizeof(T)); }
    const char *source_name(origin source) {
      return source == origin::local ? "evaluate-local" : source == origin::frame ? "frame-tag" : "global-tag";
    }
    void message(const char *text) { reshade::log::message(reshade::log::level::info, text); }
    std::uint64_t lose(loss_diagnostics::reason cause, const char *site, std::uint32_t line,
        loss_diagnostics::context details = loss_context) {
      const auto previous = loss_revision.fetch_add(1, std::memory_order_acq_rel);
      const DWORD saved_error = GetLastError();
      // A contended or overwritten diagnostic slot is simply unavailable. It
      // must never itself lose source observations or alter the revision count.
      loss_journal.publish({previous + 1, GetTickCount64(), GetCurrentThreadId(), line, cause, site, details});
      SetLastError(saved_error);
      return previous;
    }
    void invalid_observation(loss_diagnostics::reason cause, const char *site, std::uint32_t line) {
      ++invalid; lose(cause, site, line);
    }
    void dropped_observation(loss_diagnostics::reason cause, const char *site, std::uint32_t line,
        loss_diagnostics::context details = loss_context) {
      ++dropped; lose(cause, site, line, details);
    }
    bool middleware_requested() {
      return !sunshine_addon_lifetime::stopping() &&
        (requested.load(std::memory_order_acquire) || source_requested.load(std::memory_order_acquire));
    }
    unsigned tag_index(std::uint32_t type) {
      if (!requested.load(std::memory_order_acquire) && type != 0 && type != 48 && type != 53)
        return static_cast<unsigned>(observed_tag_types.size());
      for (unsigned i = 0; i != observed_tag_types.size(); ++i) if (observed_tag_types[i] == type) return i;
      return static_cast<unsigned>(observed_tag_types.size());
    }
    // Lifecycle-only table: no draw callbacks, COM/GPU calls, or nested locks.
    // These rare, bounded metadata updates must not drop a destroy event or
    // forget every existing resource merely because a reader held the table.
    void source_resource_event(std::uint64_t native, std::uint64_t lifetime, std::uint64_t device, bool add) {
      const auto generation = observation_generation.load(std::memory_order_acquire);
      if (!source_requested.load(std::memory_order_acquire) || !native || !lifetime) return;
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
      ++source_lifecycle_waiters;
#endif
      AcquireSRWLockExclusive(&source_resources_lock);
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
      --source_lifecycle_waiters;
#endif
      if (!source_requested.load(std::memory_order_acquire) ||
          generation != observation_generation.load(std::memory_order_acquire)) {
        ReleaseSRWLockExclusive(&source_resources_lock); return;
      }
      if (source_resources_lost.exchange(false, std::memory_order_acq_rel)) source_resources.fill({});
      source_resource *empty = nullptr;
      for (auto &entry : source_resources) {
        if (!entry.native && !empty) empty = &entry;
        if (entry.native != native) continue;
        if (add) entry = {native, lifetime, device};
        else if (entry.lifetime == lifetime) entry = {};
        ReleaseSRWLockExclusive(&source_resources_lock); return;
      }
      if (add && empty) *empty = {native, lifetime, device};
      else if (add) {
        source_resources_lost.store(true, std::memory_order_release);
        dropped_observation(loss_diagnostics::reason::resource_table_full, __func__, __LINE__, {});
      }
      ReleaseSRWLockExclusive(&source_resources_lock);
    }
    source_resource source_identity(std::uint64_t native) {
      source_resource result;
      const auto generation = observation_generation.load(std::memory_order_acquire);
      if (!source_requested.load(std::memory_order_acquire) || !native) return result;
      AcquireSRWLockShared(&source_resources_lock);
      if (source_requested.load(std::memory_order_acquire) &&
          generation == observation_generation.load(std::memory_order_acquire) &&
          !source_resources_lost.load(std::memory_order_acquire))
        for (const auto &entry : source_resources) if (entry.native == native) { result = entry; break; }
      ReleaseSRWLockShared(&source_resources_lock);
      return result;
    }
    std::uint64_t command_ticket() {
      const auto ticket = observation_generation.load(std::memory_order_acquire);
      return !sunshine_addon_lifetime::stopping() && requested.load(std::memory_order_acquire) &&
        observation_generation.load(std::memory_order_acquire) == ticket ? ticket : 0;
    }
    bool current_commands(std::uint64_t ticket) {
      return !sunshine_addon_lifetime::stopping() && ticket && requested.load(std::memory_order_acquire) &&
        observation_generation.load(std::memory_order_acquire) == ticket;
    }
    bool content_requested() {
      if (sunshine_addon_lifetime::stopping()) return false;
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
      if (test_content_requested.load(std::memory_order_acquire)) return true;
#endif
      return requested.load(std::memory_order_acquire) && observing.load(std::memory_order_acquire);
    }
    void sync_content_observation() {
      const bool active = content_requested();
      if (active != content_observing) {
        content_ledger.restart_observations();
        content_observing = active;
        native_discard::set_active(active);
      }
    }
    template<class Function> void command_event(Function &&function) {
      const auto ticket = command_ticket();
      if (!ticket) return;
      if (!TryAcquireSRWLockExclusive(&commands_lock)) {
        command_observation_lost.store(true, std::memory_order_release);
        dropped_observation(loss_diagnostics::reason::commands_busy, __func__, __LINE__);
        return;
      }
      if (current_commands(ticket)) {
        // A missed lifecycle callback cannot be reconstructed from native pointer
        // equality. Require fresh lifecycle observations after a tracking loss.
        if (command_observation_lost.exchange(false, std::memory_order_acq_rel)) {
          command_tracker.clear();
          content_ledger.clear();
        }
        sync_content_observation();
        function(command_tracker);
      }
      ReleaseSRWLockExclusive(&commands_lock);
    }
    void native_discard_invalidated(std::uint64_t command, std::uint64_t resource) {
      if (!content_requested()) return;
      if (!command) {
        command_event([](commands::tracker &) { content_ledger.restart_observations(); });
      } else depth_content_invalidate(command, resource);
    }
    bool recording_unchanged(const commands::recording_marker &marker) {
      const auto ticket = command_ticket();
      if (!ticket || !TryAcquireSRWLockShared(&commands_lock)) return false;
      const bool same = !command_observation_lost.load(std::memory_order_acquire) &&
        command_tracker.same_open_recording(marker, loss_revision.load(std::memory_order_acquire));
      ReleaseSRWLockShared(&commands_lock);
      return same && current_commands(ticket);
    }
    [[maybe_unused]] commands::association associate_commands(const commands::recording_marker &evaluation,
        const commands::recording_marker &copy, std::uint64_t queue) {
      commands::association out;
      const auto ticket = command_ticket();
      if (!ticket) return out;
      if (!TryAcquireSRWLockShared(&commands_lock)) { out.state = commands::status::busy; return out; }
      const auto loss = loss_revision.load(std::memory_order_acquire);
      out = command_tracker.associate(evaluation, copy, queue, loss);
      if (command_observation_lost.load(std::memory_order_acquire)) out.state = commands::status::lost_observation;
      ReleaseSRWLockShared(&commands_lock);
      if (!current_commands(ticket) || loss != loss_revision.load(std::memory_order_acquire))
        out.state = commands::status::lost_observation;
      return out;
    }
    void capture_content(evaluation_snapshot &out) {
      const auto ticket = command_ticket();
      if (!ticket) return;
      if (!TryAcquireSRWLockExclusive(&commands_lock)) {
        auto details = loss_context;
        details.sequence = out.sequence; details.viewport = out.viewport; details.feature = out.feature;
        dropped_observation(loss_diagnostics::reason::commands_busy, __func__, __LINE__, details); return;
      }
      if (current_commands(ticket) && !command_observation_lost.load(std::memory_order_acquire)) {
        sync_content_observation();
        const auto loss = loss_revision.load(std::memory_order_acquire);
        // Tags are immutable copies by now. Atomically capture the command marker
        // and all content uses before entering the game's original evaluation.
        out.evaluation_recording = command_tracker.mark(out.command_buffer, loss);
        const auto device = command_tracker.recording_device(out.evaluation_recording);
        for (auto &tag : out.tags) if (content_observing && tag.present && tag.supported && tag.value.native_resource)
          tag.content_use = content_ledger.use(out.evaluation_recording, tag.value.native_resource, device);
      }
      ReleaseSRWLockExclusive(&commands_lock);
    }
    void finish_content(evaluation_snapshot &value) {
      const auto ticket = command_ticket();
      if (!ticket || !TryAcquireSRWLockShared(&commands_lock)) {
        for (auto &tag : value.tags) tag.content_use.state = content::status::lost_observation;
        return;
      }
      const bool valid = current_commands(ticket) && !command_observation_lost.load(std::memory_order_acquire) &&
        content_observing && content_requested() &&
        command_tracker.same_open_recording(value.evaluation_recording, loss_revision.load(std::memory_order_acquire));
      for (auto &tag : value.tags) if (tag.content_use.valid())
        tag.content_use.state = valid ? content_ledger.unchanged(tag.content_use) : content::status::lost_observation;
      ReleaseSRWLockShared(&commands_lock);
    }
    [[maybe_unused]] content::association associate_content(const content::use_snapshot &use, const content::copy_snapshot &copy,
        content::resource_key source, content::resource_key backup) {
      content::association out;
      const auto ticket = command_ticket();
      if (!ticket || !TryAcquireSRWLockShared(&commands_lock)) { out.state = content::status::lost_observation; return out; }
      const auto loss = loss_revision.load(std::memory_order_acquire);
      out = content_ledger.associate(use, copy, source, backup);
      if (!content_observing || !content_requested() || command_observation_lost.load(std::memory_order_acquire) ||
          use.recording.loss != loss || copy.recording.loss != loss)
        out.state = content::status::lost_observation;
      ReleaseSRWLockShared(&commands_lock);
      if (!current_commands(ticket) || loss != loss_revision.load(std::memory_order_acquire)) out.state = content::status::lost_observation;
      return out;
    }
    void associate_evidence(evaluation_snapshot &out, const selected_depth &selected) {
      const auto ticket = command_ticket();
      if (!ticket || !TryAcquireSRWLockShared(&commands_lock)) {
        out.command_association.state = commands::status::busy;
        return;
      }
      const auto loss = loss_revision.load(std::memory_order_acquire);
      // One lock gives the combined command/content result one observation point.
      // A destroy/reset between two independent queries must not splice evidence.
      out.command_association = command_tracker.associate(out.evaluation_recording, selected.capture_marker, selected.command_queue, loss);
      if (out.evaluation_recording && !out.recording_stable) out.command_association.state = commands::status::recording_changed;
      if (out.command_association.associated()) {
        out.status = evidence_status::command_associated_evaluation;
        if (content_observing && content_requested() && content::same_recording(selected.capture_marker, selected.depth_copy.recording) &&
            selected.capture_marker.event == selected.depth_copy.recording.event)
          out.content_association = content_ledger.associate(out.tags[out.matched_tag].content_use, selected.depth_copy,
            {selected.resource, selected.source_id}, {selected.backup_resource, selected.backup_id});
        if (out.content_association.matched()) out.status = evidence_status::tracked_content_evaluation;
      }
      if (command_observation_lost.load(std::memory_order_acquire)) out.status = evidence_status::observation_lost;
      ReleaseSRWLockShared(&commands_lock);
      if (!current_commands(ticket) || loss != loss_revision.load(std::memory_order_acquire)) out.status = evidence_status::observation_lost;
    }
    std::uint64_t observation_ticket() {
      const auto ticket = observation_generation.load(std::memory_order_acquire);
      if (!middleware_requested() || !observing.load(std::memory_order_acquire)) return 0;
      return observation_generation.load(std::memory_order_acquire) == ticket ? ticket : 0;
    }
    bool current(std::uint64_t ticket) {
      return ticket && middleware_requested() && observing.load(std::memory_order_acquire) &&
        observation_generation.load(std::memory_order_acquire) == ticket;
    }

    metadata_owner::read_pin read_metadata() {
      auto pin = metadata.read();
      if (pin && (pin->epoch != epoch.load(std::memory_order_acquire) ||
          pin->observation != observation_generation.load(std::memory_order_acquire))) pin.reset();
      return pin;
    }
    metadata_owner::transaction write_metadata(std::uint64_t ticket) {
      auto update = metadata.try_write();
      if (update) {
        if (!current(ticket)) update.reset();
        else { update->epoch = epoch.load(std::memory_order_acquire); update->observation = ticket; }
      }
      return update;
    }
    void metadata_write_lost(const metadata_owner::transaction &update, const char *site,
        std::uint32_t line, loss_diagnostics::context details = loss_context) {
      dropped_observation(update.failure() == observation::write_failure::storage_busy ?
        loss_diagnostics::reason::metadata_storage_busy : loss_diagnostics::reason::records_busy, site, line, details);
    }
    viewport_record &slot(metadata_state &state, std::uint32_t viewport) {
      for (auto &record : state.records) if (record.used && record.viewport == viewport) return record;
      auto *chosen = &state.records[0];
      for (auto &record : state.records) {
        if (!record.used) { chosen = &record; break; }
        if (record.changed < chosen->changed) chosen = &record;
      }
      *chosen = {};
      chosen->used = true;
      chosen->viewport = viewport;
      return *chosen;
    }
    bool same_frame(const frame_identity &a, const frame_identity &b) {
      return a.kind != frame_identity_kind::unavailable && a.kind == b.kind &&
        (a.kind == frame_identity_kind::v1_numeric ? a.numeric == b.numeric :
          a.generation == b.generation && a.token == b.token);
    }
    void lose_presentation() { ++presentation_loss; }
    void drop_presentation() { ++presentation_dropped; lose_presentation(); }
    presentation_record *presentation_thread(bool create) {
      const auto id = GetCurrentThreadId();
      presentation_record *empty{};
      for (auto &record : presentations) {
        // A retained kernel thread handle distinguishes a dead/reused OS ID.
        // No TLS allocation or unbounded per-thread container is used in hooks.
        if (record.thread) {
          const auto state = WaitForSingleObject(record.thread, 0);
          if (state == WAIT_OBJECT_0 || state == WAIT_FAILED) {
            if (state == WAIT_FAILED) drop_presentation();
            CloseHandle(record.thread); record = {};
          }
        }
        if (record.thread && record.value.thread_id == id) return &record;
        if (!record.thread && !empty) empty = &record;
      }
      if (!create || !empty) return nullptr;
      HANDLE thread{};
      if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &thread,
          SYNCHRONIZE, FALSE, 0)) return nullptr;
      empty->thread = thread;
      empty->value.thread_id = id;
      empty->value.thread_generation = ++presentation_generation;
      return empty;
    }
    frame_identity numeric_frame(std::uint32_t frame) {
      return {frame_identity_kind::v1_numeric, 0, frame, 0, true};
    }
    bool token_current(const metadata_state &state, const frame_identity &frame) {
      // Constants-call identity is checked in the same immutable metadata
      // version; token mint identity has a separate bounded owner.
      if (frame.kind == frame_identity_kind::v1_numeric) return true;
      if (frame.kind != frame_identity_kind::v2_constants_call &&
          frame.kind != frame_identity_kind::v2_observed_token) return false;
      auto pin = token_metadata.read();
      if ((!pin && token_metadata.has_publication()) ||
          (pin && pin->observation != observation_generation.load(std::memory_order_acquire))) return false;
      bool current_token = frame.kind == frame_identity_kind::v2_constants_call;
      if (pin) for (const auto &token : pin->tokens) {
        if (frame.kind == frame_identity_kind::v2_constants_call) {
          if (token.frame.token == frame.token && token.frame.generation > frame.generation) {
            current_token = false; break;
          }
        } else if (same_frame(token.frame, frame)) {
          current_token = true; break;
        }
      }
      if (!current_token) return false;
      // Only a synchronous global FG tag may use this latest successful
      // constants-call identity. It is never an SDK frame number/token mint.
      if (frame.kind == frame_identity_kind::v2_constants_call) {
        for (const auto &record : state.records)
          if (record.used && same_frame(record.camera_frame, frame)) return true;
        return false;
      }
      return true;
    }
    bool presentation_token_current(const frame_identity &frame) {
      // Presentation markers carry minted/numeric identities, never the
      // synchronous FG-only constants-call association.
      if (frame.kind == frame_identity_kind::v2_constants_call) return false;
      static const metadata_state empty;
      return token_current(empty, frame);
    }
    frame_identity observed_frame(std::uintptr_t address) {
      frame_identity result;
      auto pin = token_metadata.read();
      if (!pin) {
        if (token_metadata.has_publication()) dropped_observation(loss_diagnostics::reason::tokens_busy, __func__, __LINE__);
        return result;
      }
      if (pin->observation != observation_generation.load(std::memory_order_acquire)) return result;
      for (const auto &token : pin->tokens) if (token.frame.token == address && address) { result = token.frame; break; }
      return result;
    }
    frame_identity presentation_frame(std::uintptr_t address) {
      frame_identity result;
      auto pin = token_metadata.read();
      if (!pin) { if (token_metadata.has_publication()) drop_presentation(); return result; }
      if (pin->observation != observation_generation.load(std::memory_order_acquire)) return result;
      for (const auto &token : pin->tokens) if (token.frame.token == address && address) { result = token.frame; break; }
      return result;
    }
    std::uint64_t start_presentation_marker(std::uint32_t marker, const frame_identity &frame, std::uint64_t ticket) {
      if ((marker != 4 && marker != 5) || !current(ticket)) return 0;
      const auto serial = ++call_sequence;
      if (!TryAcquireSRWLockExclusive(&presentations_lock)) { drop_presentation(); return 0; }
      if (!current(ticket)) { ReleaseSRWLockExclusive(&presentations_lock); return 0; }
      auto *record = presentation_thread(true);
      if (!record) { ReleaseSRWLockExclusive(&presentations_lock); drop_presentation(); return 0; }
      auto &value = record->value;
      const auto prior = value.status;
      record->pending_sequence = serial;
      value.last_marker = marker;
      value.explicit_bracket = false; // End entry closes visibility before original can reenter.
      value.epoch = epoch;
      value.loss_revision = loss_revision.load(std::memory_order_acquire);
      value.marker_loss_revision = presentation_loss.load(std::memory_order_acquire);
      if (!presentation_token_current(frame)) value.status = presentation_status::untracked_frame;
      else if (marker == 4) {
        const auto &previous = record->latest_started;
        const bool repeated = same_frame(previous, frame);
        const bool older = previous.kind == frame.kind && previous.kind != frame_identity_kind::unavailable &&
          (frame.kind == frame_identity_kind::v1_numeric ? frame.numeric <= previous.numeric : frame.generation <= previous.generation);
        value.frame = frame;
        value.start_sequence = serial; value.end_sequence = 0;
        value.start_tick = GetTickCount64(); value.end_tick = 0;
        value.status = repeated ? presentation_status::repeated_frame : older || prior == presentation_status::open_bracket ||
            prior == presentation_status::pending_marker ? presentation_status::out_of_order : presentation_status::pending_marker;
        if (!older) record->latest_started = frame;
      } else {
        value.end_sequence = serial; value.end_tick = GetTickCount64();
        value.status = prior == presentation_status::open_bracket && same_frame(value.frame, frame) ?
          presentation_status::pending_marker : presentation_status::out_of_order;
      }
      ReleaseSRWLockExclusive(&presentations_lock);
      return serial;
    }
    void finish_presentation_marker(std::uint32_t marker, std::uint64_t serial, bool success, std::uint64_t ticket) {
      if (!serial || !current(ticket)) return;
      if (!TryAcquireSRWLockExclusive(&presentations_lock)) { drop_presentation(); return; }
      auto *record = current(ticket) ? presentation_thread(false) : nullptr;
      if (record && record->pending_sequence == serial) {
        auto &value = record->value;
        if (!success) { value.status = presentation_status::failed_marker; value.explicit_bracket = false; lose_presentation(); }
        else if (value.marker_loss_revision != presentation_loss.load(std::memory_order_acquire) ||
            value.loss_revision != loss_revision.load(std::memory_order_acquire)) value.status = presentation_status::observation_lost;
        else if (value.status == presentation_status::pending_marker && !presentation_token_current(value.frame)) value.status = presentation_status::untracked_frame;
        else if (value.status == presentation_status::pending_marker) {
          value.status = marker == 4 ? presentation_status::open_bracket : presentation_status::ended_bracket;
          value.explicit_bracket = marker == 4;
        }
      }
      ReleaseSRWLockExclusive(&presentations_lock);
    }
    void queue_pcl_target(void *entry, std::uint64_t ticket) {
      if (!entry) { current_pcl_target = nullptr; pcl_available = false; lose_presentation(); return; }
      if (!TryAcquireSRWLockExclusive(&pcl_lock)) { current_pcl_target = nullptr; pcl_available = false; drop_presentation(); return; }
      if (!current(ticket)) { ReleaseSRWLockExclusive(&pcl_lock); return; }
      if (current_pcl_target.exchange(entry, std::memory_order_acq_rel) != entry) { pcl_available = false; lose_presentation(); }
      for (unsigned i = 0; i != pcl_target_count; ++i) if (pcl_targets[i].entry == entry) {
        pcl_available.store(pcl_targets[i].state.load(std::memory_order_acquire) == 1, std::memory_order_release);
        ReleaseSRWLockExclusive(&pcl_lock); return;
      }
      if (pcl_target_count == pcl_targets.size()) {
        pcl_available = false; ReleaseSRWLockExclusive(&pcl_lock); drop_presentation(); return;
      }
      MEMORY_BASIC_INFORMATION memory{};
      HMODULE owner{};
      const bool executable = VirtualQuery(entry, &memory, sizeof(memory)) == sizeof(memory) && memory.State == MEM_COMMIT &&
        memory.Type == MEM_IMAGE && (memory.Protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)) &&
        !(memory.Protect & (PAGE_GUARD | PAGE_NOACCESS));
      if (!executable || !GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, reinterpret_cast<LPCWSTR>(entry), &owner)) {
        pcl_available = false; ReleaseSRWLockExclusive(&pcl_lock); drop_presentation(); return;
      }
      auto &slot = pcl_targets[pcl_target_count++];
      slot.entry = entry; slot.owner = owner;
      pcl_available = false;
      ReleaseSRWLockExclusive(&pcl_lock);
    }
    template<unsigned Index> std::int32_t hook_pcl_marker(std::uint32_t marker, const abi_v2::frame_token &frame) {
      const DWORD incoming = GetLastError();
      const auto ticket = observation_ticket();
      const bool sample = ticket && pcl_available.load(std::memory_order_acquire) &&
        current_pcl_target.load(std::memory_order_acquire) == pcl_targets[Index].entry;
      if (ticket) ++pcl_calls;
      const auto identity = sample && (marker == 4 || marker == 5) ? presentation_frame(reinterpret_cast<std::uintptr_t>(&frame)) : frame_identity{};
      const auto serial = sample ? start_presentation_marker(marker, identity, ticket) : 0;
      SetLastError(incoming);
      const auto result = pcl_targets[Index].original.load(std::memory_order_acquire)(marker, frame);
      const DWORD outgoing = GetLastError();
      if (serial) finish_presentation_marker(marker, serial, result == 0, ticket);
      SetLastError(outgoing);
      return result;
    }
    constexpr std::array<abi_v2::pcl_set_marker, 4> pcl_hooks{hook_pcl_marker<0>, hook_pcl_marker<1>, hook_pcl_marker<2>, hook_pcl_marker<3>};
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
        same(copy.base.type, fg_options_guid) && copy.base.version >= 1 && copy.base.version <= 3 &&
        !copy.base.next && copy.mode <= 2 && (!copy.mode || copy.generated_frames != 0);
      SetLastError(incoming);
      const auto options_loss = fg_loss_revision.load(std::memory_order_acquire);
      ++fg_forwarding_depth;
      const auto result = fg_targets[Index].original.load(std::memory_order_acquire)(viewport, options);
      --fg_forwarding_depth;
      const DWORD outgoing = GetLastError();
      std::uint64_t retired_epoch{};
      if (sample) dump_metadata::observe_sl_fg(dump_stamp(ticket, serial, GetTickCount64(), valid_viewport ? id : UINT32_MAX), copy.mode, copy.generated_frames, valid, result == 0);
      bool changed = false;
      auto update = sample && current(ticket) ? write_metadata(ticket) : metadata_owner::transaction{};
      if (update) {
        if (current(ticket)) {
          if (!valid_viewport) {
            for (auto &record : update->records) record.frame_generation = {};
          } else {
            auto &state = slot(*update, id).frame_generation;
            if (serial >= state.sequence) {
              changed = !state.sequence || state.known != (valid && result == 0) ||
                state.mode != copy.mode || state.generated_frames != copy.generated_frames;
              state = {update->epoch, serial, GetTickCount64(), options_loss, id, copy.mode, copy.generated_frames,
                valid && result == 0, valid && result == 0 && copy.mode != 0, valid && copy.mode == 2};
              if (valid && result == 0 && copy.mode == 0) retired_epoch = update->epoch;
            }
          }
        }
        if (current(ticket) && !update.commit()) ++fg_loss_revision;
      } else if (sample && current(ticket)) ++fg_loss_revision;
      update.reset(); // No mutation ownership across capture/COM reentry below.
      if (retired_epoch) depth_capture::retire_source(sunshine_scene_depth::provider_kind::streamline,
        retired_epoch, (1ull << 63) | id);
      if (sample && current(ticket) && (!valid || result != 0 || retired_epoch)) {
        if (valid_viewport) sunshine_game3d::ui_mask::invalidate_scope(epoch, id);
        else sunshine_game3d::ui_mask::invalidate_all();
      }
      if (changed) {
        char text[256]{};
        std::snprintf(text, sizeof(text), "Sunshine Streamline frame generation: viewport=%u mode=%u generated_frames=%u supported=%u success=%u",
          id, copy.mode, copy.generated_frames, valid, result == 0);
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
      if (!ticket || installed_abi != abi::v2_7_30 || !original_feature_function_v2) return;
      // Engines may cache this public function before the first ReShade poll.
      // Resolve its real entry at discovery so those cached pointers also pass
      // through our deferred detour. A lookup never establishes enabled state.
      const DWORD saved = GetLastError();
      void *entry{};
      if (original_feature_function_v2(1000, "slDLSSGSetOptions", entry) == 0 && current(ticket) && entry)
        retain_fg_target(entry, ticket);
      SetLastError(saved);
    }
    std::int32_t hook_get_feature_function(std::uint32_t feature, const char *function_name, void *&function) {
      const DWORD incoming = GetLastError();
      const auto ticket = observation_ticket();
      constexpr char expected[] = "slPCLSetMarker";
      char copied_name[sizeof(expected)]{};
      const bool wanted = ticket && feature == abi_v2::feature_pcl &&
        read_bytes(function_name, copied_name, sizeof(copied_name)) && std::memcmp(copied_name, expected, sizeof(expected)) == 0;
      constexpr char fg_expected[] = "slDLSSGSetOptions";
      char copied_fg_name[sizeof(fg_expected)]{};
      const bool fg_wanted = ticket && feature == 1000 &&
        read_bytes(function_name, copied_fg_name, sizeof(copied_fg_name)) && std::memcmp(copied_fg_name, fg_expected, sizeof(fg_expected)) == 0;
      if (ticket) ++pcl_getter_calls;
      SetLastError(incoming);
      const auto result = original_feature_function_v2(feature, function_name, function);
      const DWORD outgoing = GetLastError();
      if (wanted && current(ticket)) {
        void *entry{};
        if (result == 0 && read_bytes(&function, &entry, sizeof(entry))) queue_pcl_target(entry, ticket);
        else { current_pcl_target = nullptr; pcl_available = false; lose_presentation(); }
      }
      if (fg_wanted && current(ticket)) {
        void *entry{};
        if (result == 0 && read_bytes(&function, &entry, sizeof(entry))) {
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
    bool install_pcl_hooks() {
      if (!observing.load(std::memory_order_acquire) || !middleware_requested()) return false;
      std::array<unsigned, 4> pending{}; unsigned count{};
      if (!TryAcquireSRWLockShared(&pcl_lock)) { drop_presentation(); return false; }
      for (unsigned i = 0; i != pcl_target_count; ++i)
        if (pcl_targets[i].state.load(std::memory_order_acquire) == 0) pending[count++] = i;
      ReleaseSRWLockShared(&pcl_lock);
      bool okay = true;
      for (unsigned i = 0; i != count; ++i) {
        const auto index = pending[i]; auto &slot = pcl_targets[index];
        HMODULE self{}, owner{}; void *trampoline{};
        const bool pinned = GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
            reinterpret_cast<LPCWSTR>(pcl_hooks[index]), &self) &&
          GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
            reinterpret_cast<LPCWSTR>(slot.entry), &owner) && owner == slot.owner;
        if (!pinned || MH_CreateHook(slot.entry, reinterpret_cast<void *>(pcl_hooks[index]), &trampoline) != MH_OK) {
          slot.state = 2; okay = false; lose_presentation(); continue;
        }
        slot.original.store(reinterpret_cast<abi_v2::pcl_set_marker>(trampoline), std::memory_order_release);
        if (MH_EnableHook(slot.entry) != MH_OK) { slot.state = 2; okay = false; lose_presentation(); continue; }
        slot.state.store(1, std::memory_order_release);
        lose_presentation();
        if (current_pcl_target.load(std::memory_order_acquire) == slot.entry) pcl_available = true;
      }
      return okay;
    }
    frame_record *frame_slot(metadata_state &state, std::uint32_t viewport, const frame_identity &identity, std::uint64_t serial, bool create) {
      if (identity.kind == frame_identity_kind::unavailable) return nullptr;
      for (auto &frame : state.frames) if (frame.used && frame.viewport == viewport && same_frame(frame.frame, identity)) return &frame;
      if (!create) return nullptr;
      auto *chosen = &state.frames[0];
      for (auto &frame : state.frames) {
        if (!frame.used) { chosen = &frame; break; }
        if (frame.changed < chosen->changed) chosen = &frame;
      }
      *chosen = {};
      chosen->used = true;
      chosen->viewport = viewport;
      chosen->frame = identity;
      chosen->changed = serial;
      return chosen;
    }
    const frame_record *find_frame(const metadata_state &state, std::uint32_t viewport, const frame_identity &identity) {
      for (const auto &frame : state.frames)
        if (frame.used && frame.viewport == viewport && same_frame(frame.frame, identity)) return &frame;
      return nullptr;
    }
    void store_camera(std::uint32_t viewport, const camera_data &camera, decode_status decoded,
        std::uint64_t frame, bool explicit_frame, std::uintptr_t token, std::uint64_t ticket,
        const frame_identity &identity, std::uint64_t serial, std::uint64_t loss, std::uint64_t entry_tick) {
      if (!current(ticket)) return;
      const bool invalid_camera = decoded != decode_status::ok || camera.reset > 1 ||
        (explicit_frame && camera.not_rendering_game_frames != 0) || !validate(camera).valid();
      auto details = loss_context;
      details.sequence = serial; details.viewport = viewport;
      details.reset = decoded == decode_status::ok ? camera.reset : UINT32_MAX;
      auto update = write_metadata(ticket);
      if (!update) { if (current(ticket)) metadata_write_lost(update, __func__, __LINE__, details); return; }
      auto &state = *update;
      auto &record = slot(state, viewport);
      if (serial < record.camera_serial) return;
      if (invalid_camera) lose(loss_diagnostics::reason::invalid_camera, __func__, __LINE__, details);
      else if (camera.reset == 1) {
        // Reset ends temporal history, not the validity of this frame's depth
        // or projection. Retire prior leases immediately, then associate this
        // camera with its own new revision. Never heal unrelated observation
        // loss that occurred while the original constants call was in flight.
        const auto previous_loss = lose(loss_diagnostics::reason::camera_reset, __func__, __LINE__, details);
        if (previous_loss == loss) loss = previous_loss + 1;
      }
      const auto camera_identity = identity.kind == frame_identity_kind::unavailable &&
          installed_abi == abi::v2_7_30 && token ?
        frame_identity{frame_identity_kind::v2_constants_call, serial, 0, token, false} : identity;
      record.camera = camera;
      record.decoded = decoded;
      record.key = {viewport, state.epoch, frame, explicit_frame};
      record.camera_token = token;
      record.camera_frame = camera_identity;
      record.camera_tick = entry_tick;
      record.camera_serial = serial;
      record.has_camera = true;
      record.changed = ++state.revision;
      if (token_current(state, camera_identity)) if (auto *saved = frame_slot(state, viewport, camera_identity, serial, true)) {
        if (serial >= saved->camera_serial) {
          saved->camera = camera;
          saved->decoded = decoded;
          saved->camera_serial = saved->changed = serial;
          saved->camera_tick = record.camera_tick;
          saved->loss = loss;
          saved->has_camera = true;
        }
      }
      if (current(ticket) && !update.commit()) metadata_write_lost(update, __func__, __LINE__, details);
    }
    void store_tags(batch &values) {
      if (!values.valid_viewport || !current(values.observation)) return;
      auto details = loss_context;
      details.sequence = values.serial; details.viewport = values.viewport;
      auto update = write_metadata(values.observation);
      if (!update) { if (current(values.observation)) metadata_write_lost(update, __func__, __LINE__, details); return; }
      auto &state = *update;
      for (const auto &tag : values.tags) {
        if (!tag.present) continue;
        auto tagged = tag.value;
        tagged.observation_epoch = state.epoch;
        // UntilEvaluate is valid at the synchronous copy boundary, but never
        // promoted to UntilPresent in stored/public metadata. Preserve all of
        // match()'s other identity/extent checks using a local validation copy.
        auto at_evaluation = tagged;
        if (at_evaluation.lifecycle == 0 || at_evaluation.lifecycle == 2) at_evaluation.lifecycle = 1;
        if (tag_index(tagged.type) < 3 && (!tag.supported || !tagged.native_resource ||
            ((tagged.type == 0 || tagged.type == 48) &&
              match({values.viewport, state.epoch, 0, false}, at_evaluation) != match_status::same_epoch_only))) {
          lose(loss_diagnostics::reason::tag_replaced, __func__, __LINE__, details); break;
        }
      }
      auto &record = slot(state, values.viewport);
      for (unsigned i = 0; i < values.count; ++i) {
        auto value = values.tags[i];
        if (!value.present) continue;
        value.value.observation_epoch = state.epoch;
        value.tick = values.tick;
        value.serial = values.serial;
        value.loss = values.loss;
        value.frame = values.frame;
        auto &destination = value.source == origin::local ? record.local_tags : record.global_tags;
        destination[i] = value;
        if (value.source == origin::global && value.serial >= record.active_tags[i].serial) record.active_tags[i] = value;
        if (value.source == origin::frame && token_current(state, values.frame)) {
          if (auto *saved = frame_slot(state, values.viewport, values.frame, values.serial, true)) {
            if (value.serial >= saved->tags[i].serial) saved->tags[i] = value;
            saved->changed = std::max(saved->changed, values.serial);
          }
        }
      }
      record.changed = ++state.revision;
      if (current(values.observation) && !update.commit()) metadata_write_lost(update, __func__, __LINE__, details);
    }
    evaluation_snapshot capture_evaluation(std::uint32_t viewport, std::uint32_t feature,
        const frame_identity &identity, void *commands, std::uint64_t ticket, std::uint64_t serial,
        std::uint64_t loss, const batch *local = nullptr, const metadata_state *selected_state = nullptr) {
      evaluation_snapshot out;
      out.viewport = viewport;
      out.feature = feature;
      out.frame = identity;
      out.sequence = serial;
      out.command_buffer = reinterpret_cast<std::uintptr_t>(commands);
      out.loss_revision = loss;
      out.status = evidence_status::source_associated_evaluation;
      if (!current(ticket)) { out.status = evidence_status::inactive; return out; }
      auto pin = selected_state ? metadata_owner::read_pin{} : read_metadata();
      if (!selected_state && pin) selected_state = &*pin;
      if (!selected_state && metadata.has_publication()) { out.status = evidence_status::busy; return out; }
      // The first evaluation may provide local raw depth before any constants
      // or stored tags. It still owns its explicit inputs and minted identity.
      static const metadata_state empty_state;
      if (!selected_state) selected_state = &empty_state;
      const auto &state = *selected_state;
      if (!current(ticket) || (state.observation && state.observation != ticket)) { out.status = evidence_status::inactive; return out; }
      out.epoch = state.observation ? state.epoch : epoch.load(std::memory_order_acquire);
      static const viewport_record empty;
      const viewport_record *selected = &empty;
      for (const auto &candidate : state.records)
        if (candidate.used && candidate.viewport == viewport) { selected = &candidate; break; }
      const auto &record = *selected;
      const auto *saved = find_frame(state, viewport, identity);
      if (!token_current(state, identity)) out.status = evidence_status::untracked_frame;
      else if (!saved || !saved->has_camera) out.status = evidence_status::missing_constants;
      else {
        out.camera = saved->camera;
        out.camera_tick = saved->camera_tick;
        out.camera_sequence = saved->camera_serial;
        out.decoded = saved->decoded;
        out.frame_correlated = true;
        if (saved->loss != loss) out.status = evidence_status::observation_lost;
        else if (out.decoded != decode_status::ok) out.status = evidence_status::rejected_constants;
      }
      // Freeze reset continuity with this camera/frame tuple before the native
      // evaluation. A later reset revokes older in-flight captures by revision.
      out.feedback.revision = loss + 1;
      out.feedback.reset = out.frame_correlated && out.decoded == decode_status::ok && out.camera.reset == 1;
      bool has_depth = false;
      for (unsigned i = 0; i != out.tags.size(); ++i) {
        // A standalone FG tag call supplies its complete synchronous input.
        // Older global tags must not contaminate this call's loss/validity or
        // outrank its explicit candidates before capture_fg_tag returns.
        if (feature == 1000 && local && !local->tags[i].present) continue;
        const auto *tag = &record.active_tags[i];
        if (saved && saved->tags[i].present) tag = &saved->tags[i];
        if (local && local->tags[i].present) tag = &local->tags[i];
        if (!tag->present) continue;
        const bool at_boundary = local && tag == &local->tags[i];
        auto &value = out.tags[i];
        value = {tag->value, tag->source == origin::local ? tag_scope::evaluation_local :
          tag->source == origin::frame ? tag_scope::explicit_frame : tag_scope::active_global, true, tag->supported, {}, {}, {}};
        value.value.observation_epoch = out.epoch;
        value.observed_tick = at_boundary ? local->tick : tag->tick;
        value.observation_sequence = at_boundary ? serial : tag->serial;
        value.native_state = tag->native_state;
        value.state_requires_observation = tag->state_requires_observation;
        value.direct_source = tag->direct_source;
        value.source_present_generation = tag->source_present_generation;
        value.precision_scale = tag->precision_scale;
        value.precision_bias = tag->precision_bias;
        const auto identity = source_identity(value.value.native_resource);
        // Freeze only the same lifetime seen when tagging; a recycled pointer
        // cannot acquire a fresh identity just because evaluation comes later.
        if (identity.lifetime == tag->resource_lifetime && identity.device == tag->resource_device) {
          value.resource_lifetime = identity.lifetime;
          value.resource_device = identity.device;
        }
        const auto tag_loss = at_boundary ? local->loss : tag->loss;
        if (tag_loss != loss) {
          if (out.status == evidence_status::source_associated_evaluation) out.status = evidence_status::observation_lost;
          value.supported = false;
        }
        if (i < 2 && value.supported && value.value.native_resource) has_depth = true;
      }
      for (unsigned i = 0; i != out.colors.size(); ++i) {
        const unsigned index = i + 3;
        if (feature == 1000 && local && !local->tags[index].present) continue;
        const auto *tag = &record.active_tags[index];
        if (saved && saved->tags[index].present) tag = &saved->tags[index];
        if (local && local->tags[index].present) tag = &local->tags[index];
        if (!tag->present) continue;
        auto &value = out.colors[i];
        value.value = tag->value; value.value.observation_epoch = out.epoch;
        value.scope = tag->source == origin::local ? tag_scope::evaluation_local :
          tag->source == origin::frame ? tag_scope::explicit_frame : tag_scope::active_global;
        value.present = true;
        value.supported = tag->supported && (tag->source == origin::local ? local->loss : tag->loss) == loss;
        value.identity_available = value.supported && value.value.native_resource != 0;
        value.observed_tick = tag->source == origin::local ? local->tick : tag->tick;
        value.observation_sequence = tag->source == origin::local ? serial : tag->serial;
      }
      pin.reset();
      capture_content(out);
      if (out.frame_correlated) {
        out.projection = validate(out.camera);
        if (out.decoded != decode_status::ok) out.status = evidence_status::rejected_constants;
        else if (out.camera.reset == 1 && out.projection.valid() &&
            out.status == evidence_status::source_associated_evaluation) out.status = evidence_status::camera_reset;
        else if (!out.projection.valid() || out.camera.reset > 1 ||
            (identity.kind == frame_identity_kind::v1_numeric && out.camera.not_rendering_game_frames != 0))
          out.status = evidence_status::invalid_camera;
        else if (out.status == evidence_status::source_associated_evaluation && !has_depth) out.status = evidence_status::missing_depth;
      }
      if (out.status == evidence_status::source_associated_evaluation && loss != loss_revision.load(std::memory_order_acquire))
        out.status = evidence_status::observation_lost;
      return out;
    }
    void finish_evaluation(evaluation_snapshot value, bool success, std::uint64_t ticket) {
      if (!current(ticket)) return;
      auto details = loss_context;
      details.sequence = value.sequence; details.viewport = value.viewport; details.feature = value.feature;
      details.reset = value.frame_correlated && value.decoded == decode_status::ok ? value.camera.reset : UINT32_MAX;
      value.successful_evaluation = success;
      value.tick = GetTickCount64();
      value.recording_stable = success && recording_unchanged(value.evaluation_recording);
      if (success) finish_content(value);
      if (!success) {
        value.status = evidence_status::evaluation_failed;
        lose(loss_diagnostics::reason::sdk_failure, __func__, __LINE__, details);
      }
      else if (value.status == evidence_status::source_associated_evaluation &&
          value.loss_revision != loss_revision.load(std::memory_order_acquire)) value.status = evidence_status::observation_lost;
      auto update = write_metadata(ticket);
      if (!update) { if (current(ticket)) metadata_write_lost(update, __func__, __LINE__, details); return; }
      if (current(ticket)) {
        auto &state = *update;
        auto &record = slot(state, value.viewport);
        // Lifecycle2 ends at this return even when the feature failed. Strong
        // COM references do not extend pixel validity. Never consume a newer
        // concurrent tag or turn an evaluation-local tag into a global one.
        for (unsigned i = 0; i != 2; ++i) {
          const auto &used = value.tags[i];
          if (!used.present || used.value.lifecycle != 2 || used.scope == tag_scope::evaluation_local) continue;
          tag_record *original = nullptr;
          if (used.scope == tag_scope::active_global) original = &record.active_tags[i];
          else if (auto *saved = frame_slot(state, value.viewport, value.frame, 0, false)) original = &saved->tags[i];
          if (original && original->present && original->serial == used.observation_sequence) {
            original->supported = false;
            original->direct_source.reset();
          }
        }
        // Complete calls out of order without replacing a newer evaluation by
        // an older one. The published camera/tag tuple never follows later writes.
        if (value.sequence > record.evaluation.sequence) record.evaluation = value;
        if (success) {
          record.feature = value.feature;
          record.evaluation_tick = value.tick;
          record.changed = ++state.revision;
        }
      }
      if (current(ticket) && !update.commit()) metadata_write_lost(update, __func__, __LINE__, details);
    }
    bool viewport_value(const abi_v2::viewport &value, std::uint32_t &output) {
      abi_v2::viewport copy{};
      constexpr auto prefix = offsetof(abi_v2::viewport, value) + sizeof(copy.value);
      return read_bytes(&value, &copy, prefix) && decode_v2_viewport(&copy, prefix, output) == decode_status::ok;
    }
    bool decode_precision(const void *extension, double &scale, double &bias) {
      scale = 1; bias = 0;
      if (!extension) return true;
      precision_info precision{};
      if (!read_bytes(extension, &precision, sizeof(precision)) ||
          !same(precision.base.type, precision_guid) || precision.base.version != 1 || precision.base.next) return false;
      if (precision.formula == 0) return true;
      if (precision.formula != 1 || !std::isfinite(precision.scale) || !std::isfinite(precision.bias) || precision.scale == 0) return false;
      scale = precision.scale; bias = precision.bias;
      return true;
    }
    void append_tag(batch &out, const abi_v2::resource_tag &tag, origin source, std::uintptr_t token) {
      const unsigned index = tag_index(tag.type);
      if (index == observed_tag_types.size()) return;
      abi_v2::resource resource{};
      const bool header_ok = same(tag.base.type, tag_guid) && tag.base.version == 1;
      constexpr auto resource_prefix = offsetof(abi_v2::resource, height) + sizeof(resource.height);
      const bool resource_ok = header_ok && (!tag.resource_ptr || read_bytes(tag.resource_ptr, &resource, resource_prefix));
      if (!resource_ok) {
        invalid_observation(loss_diagnostics::reason::invalid_input, __func__, __LINE__); resource = {};
      }
      tag_record value{};
      value.present = true;
      const bool precision_ok = decode_precision(tag.base.next, value.precision_scale, value.precision_bias);
      // Some validated 2.7.30 integrations zero-initialize Resource's base
      // while constructing a correctly typed ResourceTag. Accept that one
      // known layout variant, not arbitrary GUIDs/versions or extensions. The
      // native IID/device/description checks still belong to the shared owner.
      const guid empty_guid{};
      const bool zero_resource_header = installed_abi == abi::v2_7_30 && resource.base.next == nullptr &&
        same(resource.base.type, empty_guid) && resource.base.version == 0;
      const bool resource_header = resource.base.next == nullptr &&
        ((same(resource.base.type, resource_guid) && resource.base.version == 1) ||
          zero_resource_header);
      value.supported = resource_ok && precision_ok && tag.lifecycle <= 2 && (!tag.resource_ptr ||
        (resource_header && (resource.type == 0 || resource.type == 8)));
      value.source = source;
      value.token = token;
      value.value.native_resource = value.supported ? reinterpret_cast<std::uintptr_t>(resource.native) : 0;
      value.value.viewport = out.viewport;
      value.value.type = tag.type;
      value.value.lifecycle = tag.lifecycle;
      value.value.area = tag.area;
      value.value.resource_width = resource.width;
      value.value.resource_height = resource.height;
      value.native_state = resource.state;
      // A partly zero-initialized compatibility descriptor does not prove
      // COMMON. Reuse the shared observed-state path; a normal typed Resource
      // retains the SDK's declared-zero COMMON semantics.
      value.state_requires_observation = zero_resource_header && resource.state == 0;
      if (value.supported && index < 2)
        value.direct_source = depth_capture::retain_source(value.value.native_resource);
      value.source_present_generation = depth_capture::source_present_generation(value.direct_source);
      const auto identity = source_identity(value.value.native_resource);
      value.resource_lifetime = identity.lifetime;
      value.resource_device = identity.device;
      // An opaque FrameToken address can be reused. Its identity is diagnostic
      // only and must never be promoted to an exact numeric frame identifier.
      value.value.explicit_frame = false;
      if (value.supported && zero_resource_header && index < 2) {
        static std::atomic<std::uint64_t> next_compatibility_report{};
        const auto now = GetTickCount64();
        auto next = next_compatibility_report.load(std::memory_order_relaxed);
        if (now >= next && next_compatibility_report.compare_exchange_strong(next, now + 5000, std::memory_order_relaxed)) {
          char text[384]{};
          std::snprintf(text, sizeof(text),
            "Sunshine Streamline zero-base resource compatibility: tag_type=%u viewport=%u native=0x%llx type=%u state=0x%x state_proof=%s declared_extent=%ux%u native_retained=%u",
            tag.type, out.viewport, static_cast<unsigned long long>(value.value.native_resource), resource.type, resource.state,
            value.state_requires_observation ? "requires-observed-state" : "declared", resource.width, resource.height,
            value.direct_source ? 1u : 0u);
          message(text);
        }
      }
      if (!value.supported && index < 2) {
        static std::atomic<std::uint64_t> next_tag_report{};
        const auto now = GetTickCount64();
        auto next = next_tag_report.load(std::memory_order_relaxed);
        if (now >= next && next_tag_report.compare_exchange_strong(next, now + 5000, std::memory_order_relaxed)) {
          char text[640]{};
          std::snprintf(text, sizeof(text),
            "Sunshine Streamline rejected depth tag: tag_type=%u viewport=%u tag_guid=%08x tag_version=%llu tag_extension=0x%llx precision_supported=%u lifecycle=%u resource_read=%u resource_guid=%08x resource_version=%llu resource_type=%u resource_extension=0x%llx declared_native=0x%llx state=0x%x declared_extent=%ux%u",
            tag.type, out.viewport, tag.base.type.a, static_cast<unsigned long long>(tag.base.version),
            static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(tag.base.next)), precision_ok, tag.lifecycle, resource_ok,
            resource.base.type.a, static_cast<unsigned long long>(resource.base.version), resource.type,
            static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(resource.base.next)),
            static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(resource.native)), resource.state, resource.width, resource.height);
          message(text);
        }
      }
      out.tags[index] = value;
      out.count = std::max(out.count, index + 1);
    }
    batch read_tags(const abi_v2::viewport &viewport, const abi_v2::resource_tag *tags,
        std::uint32_t count, origin source, std::uintptr_t token) {
      batch out;
      out.valid_viewport = viewport_value(viewport, out.viewport);
      if (!out.valid_viewport) {
        invalid_observation(loss_diagnostics::reason::invalid_input, __func__, __LINE__); return out;
      }
      loss_context.viewport = out.viewport;
      std::array<abi_v2::resource_tag, tag_limit> copy{};
      if (count > tag_limit || (count && !read_bytes(tags, copy.data(), count * sizeof(copy[0])))) {
        invalid_observation(loss_diagnostics::reason::invalid_input, __func__, __LINE__);
        // A successful call with an unreadable/unsupported tag array must not
        // leave an older resource looking current for this known viewport.
        out.count = static_cast<unsigned>(out.tags.size());
        for (unsigned i = 0; i < out.count; ++i) {
          out.tags[i].present = true;
          out.tags[i].source = source;
          out.tags[i].value.viewport = out.viewport;
          out.tags[i].value.type = observed_tag_types[i];
        }
        return out;
      }
      for (unsigned i = 0; i < count; ++i) append_tag(out, copy[i], source, token);
      return out;
    }

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
      else if (sample && current(ticket)) lose(loss_diagnostics::reason::sdk_failure, __func__, __LINE__);
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
      if (sample && diagnostic.session) dump_metadata::observe_sl_tag_v1(diagnostic, resource, type, area);
      if (sample) {
        ++tag_calls;
        out.viewport = viewport;
        out.valid_viewport = true;
        const unsigned index = tag_index(type);
        if (index < observed_tag_types.size() && type != 53) {
          abi_v1::resource copy{};
          extent copied_area{};
          if ((!resource || read(resource, copy)) && (!area || read(area, copied_area))) {
            out.count = index + 1;
            auto &tag = out.tags[index];
            tag.present = true;
            tag.supported = copy.ext == nullptr && copy.type == 0;
            tag.value.native_resource = tag.supported ? reinterpret_cast<std::uintptr_t>(copy.native) : 0;
            tag.value.type = type;
            tag.value.viewport = viewport;
            tag.value.area = copied_area;
            tag.native_state = copy.state;
            if (tag.supported && index < 2)
              tag.direct_source = depth_capture::retain_source(tag.value.native_resource);
            tag.source_present_generation = depth_capture::source_present_generation(tag.direct_source);
            const auto identity = source_identity(tag.value.native_resource);
            tag.resource_lifetime = identity.lifetime;
            tag.resource_device = identity.device;
          } else {
            invalid_observation(loss_diagnostics::reason::invalid_input, __func__, __LINE__);
            out.count = index + 1;
            out.tags[index].present = true;
            out.tags[index].value.type = type;
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
      else if (sample && !result && current(ticket)) lose(loss_diagnostics::reason::sdk_failure, __func__, __LINE__);
      SetLastError(outgoing);
      return result;
    }
    bool direct_source_admissible(const evaluation_snapshot &snapshot) {
      // A depth tag names the renderer's source independently of whether its
      // optional camera matrices can supply a scale. Preserve the stricter
      // camera status for diagnostic queries rather than upgrading that evidence.
      switch (snapshot.status) {
        case evidence_status::source_associated_evaluation:
        case evidence_status::camera_reset:
        case evidence_status::missing_constants:
        case evidence_status::rejected_constants:
        case evidence_status::invalid_camera: break;
        default: return false;
      }
      if (!snapshot.epoch || !snapshot.sequence ||
          (snapshot.feature != 0 && !(snapshot.feature == 1000 && snapshot.tag_boundary)) ||
          snapshot.loss_revision != loss_revision.load(std::memory_order_acquire)) return false;
      if (snapshot.frame_correlated && snapshot.decoded == decode_status::ok) {
        const bool current_reset = snapshot.camera.reset == 1 && snapshot.projection.valid() &&
          snapshot.status == evidence_status::camera_reset;
        if ((snapshot.camera.reset != 0 && !current_reset) ||
            (snapshot.frame.kind == frame_identity_kind::v1_numeric && snapshot.camera.not_rendering_game_frames != 0)) return false;
      }
      auto pin = read_metadata();
      if (!pin && metadata.has_publication()) return false;
      static const metadata_state empty;
      const auto &state = pin ? *pin : empty;
      const bool associated = snapshot.epoch == (pin ? pin->epoch : epoch.load(std::memory_order_acquire)) &&
        token_current(state, snapshot.frame);
      pin.reset();
      return associated && snapshot.loss_revision == loss_revision.load(std::memory_order_acquire);
    }
    bool direct_tag_admissible(const evaluation_snapshot &snapshot, const evaluated_depth_tag &tag) {
      if (!tag.present || !tag.supported || (tag.value.type != 0 && tag.value.type != 48)) return false;
      if (snapshot.frame.kind == frame_identity_kind::v2_constants_call &&
          (!snapshot.tag_boundary || tag.scope != tag_scope::active_global || tag.value.lifecycle != 0)) return false;
      auto at_evaluation = tag.value;
      // UntilEvaluate is usable at this synchronous boundary only; never
      // promote the published lifetime. OnlyValidNow requires the tag boundary.
      if (at_evaluation.lifecycle == 2 || (snapshot.tag_boundary && at_evaluation.lifecycle == 0)) at_evaluation.lifecycle = 1;
      return match({snapshot.viewport, snapshot.epoch, 0, false}, at_evaluation) == match_status::same_epoch_only;
    }
    sunshine_scene_depth::jitter_offset depth_frame_jitter(const evaluation_snapshot &snapshot,
        const evaluated_depth_tag &tag) {
      if (!snapshot.frame_correlated || snapshot.decoded != decode_status::ok || !snapshot.camera.jitter_supplied)
        return {};
      const auto dimensions = [&](const depth_tag &source) {
        sunshine_scene_depth::jitter_offset result;
        const auto &area = source.area;
        if (source.viewport != snapshot.viewport || !source.native_resource ||
            (area.width == 0) != (area.height == 0) || (!area.width && (area.left || area.top))) return result;
        const auto width = area.width ? area.width : source.resource_width;
        const auto height = area.height ? area.height : source.resource_height;
        if (!width || !height || width > 16384 || height > 16384 ||
            (source.resource_width && (area.left > source.resource_width || width > source.resource_width-area.left)) ||
            (source.resource_height && (area.top > source.resource_height || height > source.resource_height-area.top))) return result;
        return sunshine_scene_depth::jitter_offset{snapshot.camera.jitter_offset[0], snapshot.camera.jitter_offset[1], width, height, true};
      };
      if (tag.value.type == 0) return dimensions(tag.value);
      // High-resolution depth is still jittered in render pixels. Its own
      // extent is not the denominator. Accept only a render input explicitly
      // frozen with this frame/batch; an older global tag may have another size.
      const auto &raw = snapshot.tags[0];
      const bool same_frame = raw.scope == tag_scope::explicit_frame || raw.scope == tag_scope::evaluation_local ||
        raw.observation_sequence == snapshot.sequence;
      return same_frame && direct_tag_admissible(snapshot, raw) ? dimensions(raw.value) : sunshine_scene_depth::jitter_offset{};
    }
    sunshine_scene_depth::frame normalize_depth_frame(const evaluation_snapshot &snapshot,
        const evaluated_depth_tag &tag, bool version_one) {
      sunshine_scene_depth::frame value;
      value.epoch = snapshot.epoch; value.sequence = snapshot.sequence; value.tick = GetTickCount64();
      value.observation_revision = snapshot.loss_revision;
      value.feedback = snapshot.feedback;
      value.viewport = snapshot.viewport;
      value.source_frame_generation = snapshot.frame.generation;
      value.source_frame_token = snapshot.frame.token;
      value.source_frame_numeric = snapshot.frame.numeric;
      value.source_frame_has_numeric = snapshot.frame.has_numeric;
      value.source_frame_explicit = !snapshot.tag_boundary || tag.scope == tag_scope::explicit_frame;
      value.frame_generation_input = snapshot.tag_boundary;
      value.source_id = snapshot.tag_boundary ? (1ull << 63) | snapshot.viewport : 0;
      const bool valid = (snapshot.status == evidence_status::source_associated_evaluation ||
          snapshot.status == evidence_status::camera_reset) &&
        snapshot.projection.valid() && snapshot.camera.reset <= 1 && tag.present && tag.supported &&
        tag.value.viewport == snapshot.viewport && (tag.value.type == 0 || tag.value.type == 48);
      const bool direction = snapshot.frame_correlated && snapshot.decoded == decode_status::ok &&
        snapshot.camera.depth_inverted <= 1;
      value.projection = {valid ? snapshot.projection.depth_offset : 0.0,
        valid ? snapshot.projection.depth_scale : 0.0, valid,
        direction && ((snapshot.camera.depth_inverted != 0) != (tag.precision_scale < 0)), direction};
      value.projection.raw_scale = tag.precision_scale;
      value.projection.raw_bias = tag.precision_bias;
      value.jitter = depth_frame_jitter(snapshot, tag);
      value.resource = {tag.value.native_resource, tag.value.resource_width, tag.value.resource_height,
        {tag.value.area.left, tag.value.area.top, tag.value.area.width, tag.value.area.height},
        tag.value.type == 48 ? sunshine_scene_depth::resource_kind::display_depth : sunshine_scene_depth::resource_kind::raw_depth};
      value.native_state = tag.native_state;
      value.proof = (version_one && tag.native_state == 0) || tag.state_requires_observation ? sunshine_scene_depth::state_proof::observed_nonzero :
        tag.native_state != UINT32_MAX ? sunshine_scene_depth::state_proof::declared : sunshine_scene_depth::state_proof::unavailable;
      value.valid_until = version_one || tag.value.lifecycle == 1 ? sunshine_scene_depth::lifetime::until_present :
        tag.value.lifecycle == 2 ? sunshine_scene_depth::lifetime::until_evaluation :
        snapshot.tag_boundary && tag.value.lifecycle == 0 ? sunshine_scene_depth::lifetime::at_call : sunshine_scene_depth::lifetime::unsupported;
      return value;
    }
    bool source_path_selected(const evaluation_snapshot &snapshot) {
      if (snapshot.feature != 0) return true;
      if (!fg_available.load(std::memory_order_acquire) || !current_fg_target.load(std::memory_order_acquire)) return true;
      auto pin = read_metadata();
      if (!pin) return !metadata.has_publication();
      bool selected = true;
      for (const auto &record : pin->records) if (record.used && record.viewport == snapshot.viewport) {
        const auto &fg = record.frame_generation;
        selected = !(fg.known && fg.enabled && fg.epoch == pin->epoch &&
          fg.loss_revision == fg_loss_revision.load(std::memory_order_acquire));
        break;
      }
      return selected;
    }
    std::uint64_t nominate_depth_source(const evaluation_snapshot &snapshot, bool version_one) {
      if (!source_requested.load(std::memory_order_acquire) || !depth_capture::active()) return 0;
      // FG has an explicit lifecycle for its rendered-frame inputs. The same
      // game's SR evaluations remain diagnostics and must not overwrite that
      // viewport's newer FG source with a competing renderer-input watermark.
      if (!source_path_selected(snapshot)) return 0;
      // Observation discovers interfaces; only an accepted submitted capture
      // establishes ownership in the shared capture layer.
      depth_capture::observe_provider(snapshot.command_buffer);
      const auto withdraw_invalid_attempt = [&] {
        depth_capture::begin_evaluation(snapshot.epoch, snapshot.sequence, snapshot.viewport,
          sunshine_scene_depth::provider_kind::streamline, snapshot.tag_boundary ? (1ull << 63) | snapshot.viewport : 0,
          snapshot.tag_boundary ? 0 : UINT64_MAX);
      };
      if (!direct_source_admissible(snapshot)) { withdraw_invalid_attempt(); return 0; }
      // The adapter defines semantic priority; the capture owner tries this
      // ordered set as one evaluation, including its in-progress lifetime.
      std::array<depth_capture::input, 2> candidates;
      std::size_t count = 0;
      for (const unsigned index : {1u, 0u}) {
        const auto &tag = snapshot.tags[index];
        if (!direct_tag_admissible(snapshot, tag) || !tag.direct_source) continue;
        auto &value = candidates[count];
        static_cast<sunshine_scene_depth::frame &>(value) = normalize_depth_frame(snapshot, tag, version_one);
        value.source = tag.direct_source;
        value.source_present_generation = tag.source_present_generation;
        value.force_snapshot = snapshot.tag_boundary && tag.value.lifecycle == 0;
        v1_resource_state::read_details state_details;
        const auto state = v1_resource_state::resolve(value.native_state, version_one,
          v1_private_state_ready.load(std::memory_order_acquire) ? reinterpret_cast<IUnknown *>(tag.value.native_resource) : nullptr,
          &state_details);
        if (version_one && state.proof == sunshine_scene_depth::state_proof::observed_nonzero) {
          // Bound failure reporting at the immutable evaluation boundary. An
          // enabled version adapter alone does not prove a usable state read.
          static std::atomic<std::uint64_t> next_state_report{};
          const auto now = GetTickCount64();
          auto next = next_state_report.load(std::memory_order_relaxed);
          if (now >= next && next_state_report.compare_exchange_strong(next, now + 5000, std::memory_order_relaxed)) {
            char text[640]{};
            std::snprintf(text, sizeof(text),
              "Sunshine Streamline V1 state lookup: %s; resource=0x%llx tag=%u viewport=%u sequence=%llu; %llux%u format=%u mips=%u layers=%u samples=%u; hr=0x%08lx bytes=%u chi=0x%x; tagged=0x%x; requires observed state",
              state_details.stage, static_cast<unsigned long long>(tag.value.native_resource), tag.value.type, snapshot.viewport,
              static_cast<unsigned long long>(snapshot.sequence), static_cast<unsigned long long>(state_details.description.Width),
              state_details.description.Height, unsigned(state_details.description.Format), state_details.description.MipLevels,
              state_details.description.DepthOrArraySize, state_details.description.SampleDesc.Count,
              static_cast<unsigned long>(state_details.result), state_details.bytes, state_details.encoded, value.native_state);
            message(text);
          }
        }
        value.native_state = state.native;
        if (!tag.state_requires_observation) value.proof = state.proof;
        ++count;
      }
      if (count) if (const auto ticket = depth_capture::nominate_evaluation(snapshot.command_buffer,
          candidates.data(), count, snapshot.tag_boundary ? 0 : UINT64_MAX)) return ticket;
      withdraw_invalid_attempt();
      return 0;
    }
    bool hook_evaluate_v1(void *commands, std::uint32_t feature, std::uint32_t frame, std::uint32_t viewport) {
      const DWORD incoming = GetLastError();
      sunshine_upscaler_trace::streamline_scope call_trace(feature, SUNSHINE_UPSCALER_CALLER);
      const auto ticket = observation_ticket();
      // v1 Reflex (feature 3) uses the fourth argument as a timing marker,
      // including marker 0, rather than a viewport. Only DLSS (feature 0) has
      // the renderer-input contract this probe validates. Others pass through.
      const bool sample = ticket && feature == 0;
      const loss_context_scope loss_scope(0, sample ? viewport : UINT32_MAX, feature);
      if (ticket) ++evaluation_calls;
      const auto present_serial = ticket && feature == 3 ?
        start_presentation_marker(viewport, numeric_frame(frame), ticket) : 0;
      evaluation_snapshot snapshot;
      if (sample) {
        const auto serial = ++call_sequence;
        loss_context.sequence = serial;
        snapshot = capture_evaluation(viewport, feature, numeric_frame(frame), commands, ticket,
          serial, loss_revision.load(std::memory_order_acquire));
      }
      const auto captured = sample ? nominate_depth_source(snapshot, true) : 0;
      const auto diagnostic = dump_stamp(ticket, snapshot.sequence, GetTickCount64(), viewport, 0, frame, true, reinterpret_cast<std::uint64_t>(commands));
      SetLastError(incoming);
      const bool result = original_evaluate_v1(commands, feature, frame, viewport);
      const DWORD outgoing = GetLastError();
      loss_context.sdk_result = result ? 1 : 0; loss_context.has_sdk_result = true;
      if (sample && diagnostic.session) dump_metadata::observe_sl_evaluation(diagnostic, feature, result);
      call_trace.finish(result);
      if (captured) depth_capture::finish(captured,
        result && current(ticket) && snapshot.loss_revision == loss_revision.load(std::memory_order_acquire),
        result ? depth_capture::capture_failure::evaluation_observation_changed : depth_capture::capture_failure::evaluation_failed);
      if (sample) finish_evaluation(snapshot, result, ticket);
      if (present_serial) finish_presentation_marker(viewport, present_serial, result, ticket);
      SetLastError(outgoing);
      return result;
    }
    std::int32_t hook_constants_v2(const abi_v2::constants &values, const abi_v2::frame_token &frame, const abi_v2::viewport &viewport) {
      const DWORD incoming = GetLastError();
      const auto ticket = observation_ticket();
      const bool sample = ticket != 0;
      const auto serial = sample ? ++call_sequence : 0;
      const loss_context_scope loss_scope(serial);
      const auto loss = loss_revision.load(std::memory_order_acquire);
      const auto identity = sample ? observed_frame(reinterpret_cast<std::uintptr_t>(&frame)) : frame_identity{};
      const auto entry_tick = sample ? GetTickCount64() : 0;
      auto diagnostic = dump_stamp(ticket, serial, entry_tick, UINT32_MAX, reinterpret_cast<std::uintptr_t>(&frame), identity.numeric, identity.has_numeric);
      camera_data camera;
      decode_status decoded = decode_status::short_buffer;
      std::uint32_t id{};
      bool valid_viewport = false;
      if (sample) {
        ++constant_calls;
        abi_v2::constants copy{};
        constexpr auto prefix = offsetof(abi_v2::constants, motion_vectors_jittered) + 1;
        const bool readable = read_bytes(&values, &copy, prefix);
        if (readable) decoded = decode_v2_constants(&copy, prefix, camera);
        valid_viewport = viewport_value(viewport, id);
        loss_context.viewport = valid_viewport ? id : UINT32_MAX;
        diagnostic.viewport = valid_viewport ? id : UINT32_MAX;
        if (diagnostic.session) {
          const bool tail_read = decoded == decode_status::ok && copy.base.version == 2 &&
            read_bytes(reinterpret_cast<const unsigned char *>(&values) + offsetof(abi_v2::constants, min_relative_linear_depth_object_separation),
              &copy.min_relative_linear_depth_object_separation, sizeof(copy.min_relative_linear_depth_object_separation));
          dump_metadata::observe_sl_constants(diagnostic, copy, readable, decoded, tail_read);
        }
      }
      SetLastError(incoming);
      const auto result = original_constants_v2(values, frame, viewport);
      const DWORD outgoing = GetLastError();
      loss_context.sdk_result = result; loss_context.has_sdk_result = true;
      if (diagnostic.session) dump_metadata::finish_sl_call(diagnostic, result == 0);
      if (sample && result == 0 && valid_viewport)
        store_camera(id, camera, decoded, 0, false, reinterpret_cast<std::uintptr_t>(&frame), ticket, identity, serial, loss, entry_tick);
      else if (sample && current(ticket))
        lose(result != 0 ? loss_diagnostics::reason::sdk_failure : loss_diagnostics::reason::invalid_input, __func__, __LINE__);
      SetLastError(outgoing);
      return result;
    }
    sunshine_game3d::ui_mask::attempt capture_fg_alpha(const batch &out, void *commands) {
      namespace mask = sunshine_game3d::ui_mask;
      if (!out.valid_viewport || !current(out.observation) ||
          out.loss != loss_revision.load(std::memory_order_acquire) || !out.tags[6].present) return {};
      frame_generation_snapshot fg;
      if (!query_frame_generation(out.viewport, fg) || !fg.enabled ||
          !mask::interested(fg.epoch, out.loss, out.viewport)) return {};
      // Backbuffer (53) commonly arrives in a separate color-only call, after
      // depth/HUDless tags. Its own live command/state/lifetime are authoritative;
      // no camera, depth tag, dump request, or Present correlation is required.
      const auto &tag = out.tags[6];
      depth_capture::input input;
      input.epoch = fg.epoch; input.sequence = out.serial; input.tick = out.tick;
      input.viewport = out.viewport; input.observation_revision = out.loss;
      input.source_id = (1ull << 63) | out.viewport;
      input.frame_generation_input = true;
      input.source_frame_generation = out.frame.generation;
      input.source_frame_token = tag.token;
      input.source_frame_numeric = out.frame.numeric;
      input.source_frame_has_numeric = out.frame.has_numeric;
      input.source_frame_explicit = tag.source == origin::frame;
      input.resource.native = tag.supported ? tag.value.native_resource : 0;
      input.resource.width = tag.value.resource_width; input.resource.height = tag.value.resource_height;
      input.resource.area = {tag.value.area.left, tag.value.area.top, tag.value.area.width, tag.value.area.height};
      input.native_state = tag.native_state;
      input.proof = tag.state_requires_observation ? sunshine_scene_depth::state_proof::observed_nonzero :
        sunshine_scene_depth::state_proof::declared;
      input.valid_until = !tag.supported ? sunshine_scene_depth::lifetime::unsupported : tag.value.lifecycle == 0 ?
        sunshine_scene_depth::lifetime::at_call : tag.value.lifecycle == 1 ? sunshine_scene_depth::lifetime::until_present :
        sunshine_scene_depth::lifetime::until_evaluation;
      input.force_snapshot = true;
      depth_capture::record_diagnostic retained;
      input.source = depth_capture::retain_source(input.resource.native, &retained);
      input.source_present_generation = depth_capture::source_present_generation(input.source);
      // Validate actual allocation size before recording, even if SL omitted
      // the optional resource dimensions. Partial output extents are rejected.
      if (input.source) { input.resource.width = retained.width; input.resource.height = retained.height; }
      mask::boundary where;
      where.source = input; where.command = reinterpret_cast<std::uint64_t>(commands);
      where.tag_scope = tag.source == origin::frame ? 1u : 0u;
      return mask::begin(where, input);
    }
    evaluation_snapshot capture_fg_tag(const batch &out, void *commands) {
      evaluation_snapshot snapshot;
      if (!out.valid_viewport || !source_requested.load(std::memory_order_acquire) ||
          !fg_available.load(std::memory_order_acquire) || !current_fg_target.load(std::memory_order_acquire) ||
          (!out.tags[0].present && !out.tags[1].present)) return snapshot;
      frame_identity identity = out.frame;
      auto pin = read_metadata();
      if (!pin) return snapshot;
      bool enabled = false;
      for (const auto &record : pin->records) if (record.used && record.viewport == out.viewport) {
        enabled = record.frame_generation.known && record.frame_generation.enabled && record.frame_generation.epoch == pin->epoch &&
          record.frame_generation.loss_revision == fg_loss_revision.load(std::memory_order_acquire);
        const auto &first_tag = out.tags[0].present ? out.tags[0] : out.tags[1];
        const bool synchronous_global_depth = std::any_of(out.tags.begin(), out.tags.begin() + 2, [](const tag_record &tag) {
          return tag.present && tag.source == origin::global && tag.value.lifecycle == 0;
        });
        const auto *camera_frame = find_frame(*pin, out.viewport, record.camera_frame);
        if (identity.kind == frame_identity_kind::unavailable && first_tag.source == origin::global &&
            (record.camera_frame.kind != frame_identity_kind::v2_constants_call || synchronous_global_depth) &&
            record.has_camera && record.camera_tick <= out.tick && out.tick - record.camera_tick <= 250 &&
            camera_frame && camera_frame->camera_serial < out.serial && camera_frame->loss == out.loss &&
            token_current(*pin, record.camera_frame))
          identity = record.camera_frame;
        break;
      }
      if (!enabled) return snapshot;
      snapshot = capture_evaluation(out.viewport, 1000, identity, commands,
        out.observation, out.serial, out.loss, &out, &*pin);
      snapshot.tag_boundary = true;
      if (snapshot.frame_correlated && (snapshot.camera_sequence >= out.serial || snapshot.camera_tick > out.tick ||
          out.tick - snapshot.camera_tick > 250)) {
        snapshot.status = evidence_status::stale;
        snapshot.frame_correlated = false;
      }
      return snapshot;
    }
    std::int32_t hook_tag_v2(const abi_v2::viewport &viewport, const abi_v2::resource_tag *tags, std::uint32_t count, void *commands) {
      const DWORD incoming = GetLastError();
      const auto ticket = observation_ticket();
      const bool sample = ticket != 0;
      const auto serial = sample ? ++call_sequence : 0;
      const loss_context_scope loss_scope(serial);
      const auto loss = loss_revision.load(std::memory_order_acquire);
      const auto entry_tick = sample ? GetTickCount64() : 0;
      const auto diagnostic = dump_stamp(ticket, serial, entry_tick, UINT32_MAX, 0, 0, false, reinterpret_cast<std::uint64_t>(commands));
      if (sample && diagnostic.session) dump_metadata::observe_sl_tags_v2(diagnostic, viewport, tags, count, dump_metadata::tag_scope::global);
      batch out;
      if (sample) { ++tag_calls; out = read_tags(viewport, tags, count, origin::global, 0); }
      out.observation = ticket;
      out.serial = serial;
      out.loss = loss;
      out.tick = entry_tick;
      const auto alpha = sample ? capture_fg_alpha(out, commands) : sunshine_game3d::ui_mask::attempt{};
      const auto snapshot = sample ? capture_fg_tag(out, commands) : evaluation_snapshot{};
      if (snapshot.tag_boundary) loss_context.feature = snapshot.feature;
      const auto captured = snapshot.tag_boundary ? nominate_depth_source(snapshot, false) : 0;
      SetLastError(incoming);
      const auto result = original_tag_v2(viewport, tags, count, commands);
      const DWORD outgoing = GetLastError();
      loss_context.sdk_result = result; loss_context.has_sdk_result = true;
      if (diagnostic.session) dump_metadata::finish_sl_call(diagnostic, result == 0);
      if (captured) depth_capture::finish(captured,
        result == 0 && current(ticket) && loss == loss_revision.load(std::memory_order_acquire),
        result == 0 ? depth_capture::capture_failure::evaluation_observation_changed : depth_capture::capture_failure::evaluation_failed);
      if (sample && result == 0 && out.count) store_tags(out);
      else if (sample && (result != 0 || !out.valid_viewport) && current(ticket))
        lose(result != 0 ? loss_diagnostics::reason::sdk_failure : loss_diagnostics::reason::invalid_input, __func__, __LINE__);
      if (snapshot.tag_boundary) finish_evaluation(snapshot, result == 0, ticket);
      sunshine_game3d::ui_mask::finish(alpha,
        result == 0 && current(ticket) && loss == loss_revision.load(std::memory_order_acquire));
      SetLastError(outgoing);
      return result;
    }
    std::int32_t hook_frame_tag_v2(const abi_v2::frame_token &frame, const abi_v2::viewport &viewport,
        const abi_v2::resource_tag *tags, std::uint32_t count, void *commands) {
      const DWORD incoming = GetLastError();
      const auto ticket = observation_ticket();
      const bool sample = ticket != 0;
      const auto serial = sample ? ++call_sequence : 0;
      const loss_context_scope loss_scope(serial);
      const auto loss = loss_revision.load(std::memory_order_acquire);
      const auto identity = sample ? observed_frame(reinterpret_cast<std::uintptr_t>(&frame)) : frame_identity{};
      const auto entry_tick = sample ? GetTickCount64() : 0;
      const auto diagnostic = dump_stamp(ticket, serial, entry_tick, UINT32_MAX, reinterpret_cast<std::uintptr_t>(&frame), identity.numeric, identity.has_numeric, reinterpret_cast<std::uint64_t>(commands));
      if (sample && diagnostic.session) dump_metadata::observe_sl_tags_v2(diagnostic, viewport, tags, count, dump_metadata::tag_scope::frame);
      batch out;
      if (sample) { ++tag_calls; out = read_tags(viewport, tags, count, origin::frame, reinterpret_cast<std::uintptr_t>(&frame)); }
      out.observation = ticket;
      out.serial = serial;
      out.loss = loss;
      out.frame = identity;
      out.tick = entry_tick;
      const auto alpha = sample ? capture_fg_alpha(out, commands) : sunshine_game3d::ui_mask::attempt{};
      const auto snapshot = sample ? capture_fg_tag(out, commands) : evaluation_snapshot{};
      if (snapshot.tag_boundary) loss_context.feature = snapshot.feature;
      const auto captured = snapshot.tag_boundary ? nominate_depth_source(snapshot, false) : 0;
      SetLastError(incoming);
      const auto result = original_frame_tag_v2(frame, viewport, tags, count, commands);
      const DWORD outgoing = GetLastError();
      loss_context.sdk_result = result; loss_context.has_sdk_result = true;
      if (diagnostic.session) dump_metadata::finish_sl_call(diagnostic, result == 0);
      if (captured) depth_capture::finish(captured,
        result == 0 && current(ticket) && loss == loss_revision.load(std::memory_order_acquire),
        result == 0 ? depth_capture::capture_failure::evaluation_observation_changed : depth_capture::capture_failure::evaluation_failed);
      if (sample && result == 0 && out.count) store_tags(out);
      else if (sample && (result != 0 || !out.valid_viewport) && current(ticket))
        lose(result != 0 ? loss_diagnostics::reason::sdk_failure : loss_diagnostics::reason::invalid_input, __func__, __LINE__);
      if (snapshot.tag_boundary) finish_evaluation(snapshot, result == 0, ticket);
      sunshine_game3d::ui_mask::finish(alpha,
        result == 0 && current(ticket) && loss == loss_revision.load(std::memory_order_acquire));
      SetLastError(outgoing);
      return result;
    }
    batch read_local_inputs(const base_structure **inputs, std::uint32_t count, std::uintptr_t token) {
      batch out;
      if (count > tag_limit) { invalid_observation(loss_diagnostics::reason::invalid_input, __func__, __LINE__); return out; }
      std::array<const base_structure *, tag_limit> roots{};
      if (count && !read_bytes(inputs, roots.data(), count * sizeof(roots[0]))) {
        invalid_observation(loss_diagnostics::reason::invalid_input, __func__, __LINE__); return out;
      }
      std::array<abi_v2::resource_tag, tag_limit> tags{};
      unsigned tag_count = 0, visited = 0;
      for (unsigned i = 0; i < count; ++i) {
        const auto *current = roots[i];
        while (current && visited++ < chain_limit) {
          base_structure base;
          if (!read(current, base)) { invalid_observation(loss_diagnostics::reason::invalid_input, __func__, __LINE__); return out; }
          if (same(base.type, viewport_guid)) {
            abi_v2::viewport viewport{};
            std::uint32_t id{};
            constexpr auto prefix = offsetof(abi_v2::viewport, value) + sizeof(viewport.value);
            if (!read_bytes(current, &viewport, prefix)) {
              loss_context.viewport = UINT32_MAX;
              invalid_observation(loss_diagnostics::reason::invalid_input, __func__, __LINE__); out.valid_viewport = false; return out;
            }
            // In an evaluate input list 'next' is the documented input chain,
            // which this loop independently traverses under a global cap. It
            // does not change the viewport identifier's known prefix layout.
            viewport.base.next = nullptr;
            if (
                decode_v2_viewport(&viewport, sizeof(viewport), id) != decode_status::ok ||
                (out.valid_viewport && out.viewport != id)) {
              loss_context.viewport = UINT32_MAX;
              invalid_observation(loss_diagnostics::reason::invalid_input, __func__, __LINE__); out.valid_viewport = false; return out;
            }
            out.viewport = id;
            out.valid_viewport = true;
            loss_context.viewport = id;
          } else if (same(base.type, tag_guid) && tag_count < tags.size()) {
            if (!read_bytes(current, &tags[tag_count++], sizeof(tags[0]))) {
              invalid_observation(loss_diagnostics::reason::invalid_input, __func__, __LINE__); return out;
            }
          }
          current = static_cast<const base_structure *>(base.next);
        }
        if (current) {
          loss_context.viewport = UINT32_MAX;
          invalid_observation(loss_diagnostics::reason::invalid_input, __func__, __LINE__); out.valid_viewport = false; return out;
        }
      }
      if (out.valid_viewport)
        for (unsigned i = 0; i < tag_count; ++i) append_tag(out, tags[i], origin::local, token);
      return out;
    }
    std::int32_t hook_evaluate_v2(std::uint32_t feature, const abi_v2::frame_token &frame,
        const base_structure **inputs, std::uint32_t count, void *commands) {
      const DWORD incoming = GetLastError();
      sunshine_upscaler_trace::streamline_scope call_trace(feature, SUNSHINE_UPSCALER_CALLER);
      const auto ticket = observation_ticket();
      // Non-DLSS feature inputs have different contracts; do not parse them as
      // viewport/depth metadata or let their failures revoke renderer evidence.
      const bool sample = ticket && feature == 0;
      if (ticket) ++evaluation_calls;
      const auto serial = sample ? ++call_sequence : 0;
      const loss_context_scope loss_scope(serial, UINT32_MAX, feature);
      const auto loss = loss_revision.load(std::memory_order_acquire);
      const auto identity = sample ? observed_frame(reinterpret_cast<std::uintptr_t>(&frame)) : frame_identity{};
      const auto entry_tick = sample ? GetTickCount64() : 0;
      batch out;
      if (sample) out = read_local_inputs(inputs, count, reinterpret_cast<std::uintptr_t>(&frame));
      const auto diagnostic = dump_stamp(ticket, serial, entry_tick, out.valid_viewport ? out.viewport : UINT32_MAX,
        reinterpret_cast<std::uintptr_t>(&frame), identity.numeric, identity.has_numeric, reinterpret_cast<std::uint64_t>(commands));
      if (sample && diagnostic.session) dump_metadata::observe_sl_inputs_v2(diagnostic, inputs, count);
      out.observation = ticket;
      out.serial = serial;
      out.loss = loss;
      out.frame = identity;
      out.tick = entry_tick;
      evaluation_snapshot snapshot;
      if (sample && out.valid_viewport)
        snapshot = capture_evaluation(out.viewport, feature, identity, commands, ticket, serial, loss, &out);
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
      if (entry_policy_armed) {
        entry_policy_armed = false;
        entry_policy_observed = true;
        entry_path_selected = source_path_selected(snapshot);
        entry_source_admitted = direct_source_admissible(snapshot);
      }
#endif
      const auto captured = sample && out.valid_viewport ? nominate_depth_source(snapshot, false) : 0;
      if (sample && !out.valid_viewport && source_requested.load(std::memory_order_acquire) && depth_capture::active()) {
        depth_capture::observe_provider(reinterpret_cast<std::uintptr_t>(commands));
        depth_capture::begin_evaluation(0, serial, 0xffffffffu);
      }
      SetLastError(incoming);
      const auto result = original_evaluate_v2(feature, frame, inputs, count, commands);
      const DWORD outgoing = GetLastError();
      loss_context.sdk_result = result; loss_context.has_sdk_result = true;
      if (sample && diagnostic.session) {
        dump_metadata::finish_sl_call(diagnostic, result == 0);
        dump_metadata::observe_sl_evaluation(diagnostic, feature, result == 0);
      }
      call_trace.finish(result == 0);
      if (captured) depth_capture::finish(captured,
        result == 0 && current(ticket) && snapshot.loss_revision == loss_revision.load(std::memory_order_acquire),
        result == 0 ? depth_capture::capture_failure::evaluation_observation_changed : depth_capture::capture_failure::evaluation_failed);
      if (sample && out.valid_viewport) {
        if (result == 0 && out.count) store_tags(out);
        finish_evaluation(snapshot, result == 0, ticket);
      } else if (sample && current(ticket)) {
        // An unresolvable evaluation viewport cannot leave any older camera
        // evidence looking applicable to a new successful call.
        lose(result != 0 ? loss_diagnostics::reason::sdk_failure : loss_diagnostics::reason::invalid_input, __func__, __LINE__);
      }
      SetLastError(outgoing);
      return result;
    }

    std::int32_t hook_new_token_v2(abi_v2::frame_token *&token, const std::uint32_t *frame_index) {
      const DWORD incoming = GetLastError();
      const auto ticket = observation_ticket();
      const auto serial = ticket ? ++call_sequence : 0;
      const loss_context_scope loss_scope(serial);
      std::uint32_t numeric{};
      const bool index_ok = !ticket || !frame_index || read(frame_index, numeric);
      SetLastError(incoming);
      const auto result = original_new_token_v2(token, frame_index);
      const DWORD outgoing = GetLastError();
      loss_context.sdk_result = result; loss_context.has_sdk_result = true;
      if (current(ticket)) {
        abi_v2::frame_token *returned{};
        if (result != 0 || !index_ok || !read(&token, returned) || !returned) {
          invalid_observation(result != 0 ? loss_diagnostics::reason::sdk_failure : loss_diagnostics::reason::invalid_input,
            __func__, __LINE__);
        } else {
          auto update = token_metadata.try_write();
          if (!update) {
            if (current(ticket)) dropped_observation(update.failure() == observation::write_failure::storage_busy ?
              loss_diagnostics::reason::metadata_storage_busy : loss_diagnostics::reason::tokens_busy, __func__, __LINE__);
          } else if (current(ticket)) {
            update->observation = ticket;
            const auto address = reinterpret_cast<std::uintptr_t>(returned);
            auto *chosen = &update->tokens[0];
            for (auto &candidate : update->tokens) {
              if (candidate.frame.token == address) { chosen = &candidate; break; }
              if (!candidate.frame.token || candidate.frame.generation < chosen->frame.generation) chosen = &candidate;
            }
            // A recycled address is a NEW observer generation. The optional
            // numeric index is caller-owned; absent means unknown, never zero.
            if (chosen->frame.token != address || serial > chosen->frame.generation)
              chosen->frame = {frame_identity_kind::v2_observed_token, serial, numeric, address, frame_index != nullptr};
            else lose(loss_diagnostics::reason::token_replaced, __func__, __LINE__); // Concurrent out-of-order allocation of the same address.
            if (current(ticket) && !update.commit())
              dropped_observation(loss_diagnostics::reason::metadata_storage_busy, __func__, __LINE__);
          }
        }
      }
      SetLastError(outgoing);
      return result;
    }

    bool install_hooks(abi version, const targets &functions) {
      if (version == abi::unsupported || !functions.constants || !functions.tag || !functions.evaluate ||
          (version == abi::v2_7_30 && !functions.tag_for_frame)) return false;
      if (!minhook_initialized) {
        const auto status = MH_Initialize();
        if (status != MH_OK && status != MH_ERROR_ALREADY_INITIALIZED) return false;
        minhook_initialized = true;
      }
      struct entry { void *target, *hook; void **original; };
      std::array<entry, 6> entries{};
      unsigned count{};
      if (version == abi::v1_1_1) {
        entries[count++] = {functions.constants, reinterpret_cast<void *>(hook_constants_v1), reinterpret_cast<void **>(&original_constants_v1)};
        entries[count++] = {functions.tag, reinterpret_cast<void *>(hook_tag_v1), reinterpret_cast<void **>(&original_tag_v1)};
        entries[count++] = {functions.evaluate, reinterpret_cast<void *>(hook_evaluate_v1), reinterpret_cast<void **>(&original_evaluate_v1)};
      } else {
        entries[count++] = {functions.constants, reinterpret_cast<void *>(hook_constants_v2), reinterpret_cast<void **>(&original_constants_v2)};
        entries[count++] = {functions.tag, reinterpret_cast<void *>(hook_tag_v2), reinterpret_cast<void **>(&original_tag_v2)};
        entries[count++] = {functions.tag_for_frame, reinterpret_cast<void *>(hook_frame_tag_v2), reinterpret_cast<void **>(&original_frame_tag_v2)};
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
    bool install_discard_observations() {
      if (!content_requested()) return false;
      if (!minhook_initialized) {
        const auto status = MH_Initialize();
        if (status != MH_OK && status != MH_ERROR_ALREADY_INITIALIZED) return false;
        minhook_initialized = true;
      }
      // Content mode may have changed at discovery before the next command event.
      native_discard::set_active(true);
      return native_discard::install_pending();
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
      if (installed_abi != abi::v1_1_1 || v1_private_state_checked) return;
      // The encoding is owned by sl.common, whose version can differ from the
      // interposer in a mixed installation. Resolve it once outside callbacks;
      // a delayed load remains eligible at the next existing discovery poll.
      HMODULE module{};
      if (!GetModuleHandleExW(0, L"sl.common.dll", &module)) return;
      dump_metadata::observe_module(module, "streamline", "1.1.1 resource-state adapter");
      auto info = module_version(module);
      const bool supported = versioning::classify(info) == abi::v1_1_1;
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
      dump_metadata::observe_module(module, "streamline", version == abi::v1_1_1 ? "1.1.1 observer" : version == abi::v2_7_30 ? "2.7.30 observer" : "unsupported observer ABI");
      targets functions{
        reinterpret_cast<void *>(GetProcAddress(module, "slSetConstants")),
        reinterpret_cast<void *>(GetProcAddress(module, "slSetTag")),
        reinterpret_cast<void *>(GetProcAddress(module, "slSetTagForFrame")),
        reinterpret_cast<void *>(GetProcAddress(module, "slEvaluateFeature")),
        reinterpret_cast<void *>(GetProcAddress(module, "slGetNewFrameToken")),
        reinterpret_cast<void *>(GetProcAddress(module, "slGetFeatureFunction"))};
      char identity[768]{};
      std::snprintf(identity, sizeof(identity),
        "Sunshine Streamline: version fixed-file=%u.%u.%u.%u fixed-product=%u.%u.%u.%u strings='%ls'/'%ls' result=%s; exports constants=%d tag=%d frame-tag=%d evaluate=%d new-frame-token=%d",
        HIWORD(info.file_ms), LOWORD(info.file_ms), HIWORD(info.file_ls), LOWORD(info.file_ls),
        HIWORD(info.product_ms), LOWORD(info.product_ms), HIWORD(info.product_ls), LOWORD(info.product_ls),
        info.file_text.data(), info.product_text.data(), info.reason,
        functions.constants != nullptr, functions.tag != nullptr, functions.tag_for_frame != nullptr, functions.evaluate != nullptr,
        functions.new_frame_token != nullptr);
      message(identity);
      if (version == abi::unsupported || !functions.constants || !functions.tag || !functions.evaluate ||
          (version == abi::v2_7_30 && !functions.tag_for_frame)) {
        message(version == abi::unsupported ?
          "Sunshine Streamline: camera hooks disabled because the version identity above is not validated" :
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
      const bool installed = install_hooks(version, functions);
      FreeLibrary(module);
      if (!installed) {
        observing.store(false, std::memory_order_release);
        permanently_rejected = true;
        message("Sunshine Streamline: inline-hook setup failed; retained code stays pass-through and camera observation is disabled");
        return;
      }
      observing.store(middleware_requested(), std::memory_order_release);
      message(version == abi::v1_1_1 ?
        "Sunshine Streamline: observing 1.1.1.0 constants, global depth tags and feature evaluation; direct depth capture enabled when supported" :
        "Sunshine Streamline: observing 2.7.30.0 constants, global/frame/local depth tags and feature evaluation; direct depth capture enabled when supported");
      if (version == abi::v2_7_30 && !functions.new_frame_token)
        message("Sunshine Streamline: slGetNewFrameToken unavailable; v2 evaluation frame evidence stays untracked");
    }
  }

  void initialize(HMODULE addon) {
    bool diagnostic_enabled = false;
    reshade::get_config_value(nullptr, "SUNSHINE_DEPTH", "StreamlineCameraProbe", diagnostic_enabled);
    bool source_enabled = true;
    reshade::get_config_value(nullptr, "SUNSHINE_DEPTH", "StreamlineDepthSource", source_enabled);
    bool call_trace_enabled = false;
    reshade::get_config_value(nullptr, "SUNSHINE_DEPTH", "UpscalerCallTrace", call_trace_enabled);
    bool ngx_source_enabled = true;
    reshade::get_config_value(nullptr, "SUNSHINE_DEPTH", "NGXDepthSource", ngx_source_enabled);
    bool ngx_calibration_probe = false;
    reshade::get_config_value(nullptr, "SUNSHINE_DEPTH", "NGXCalibrationProbe", ngx_calibration_probe);
    sunshine_upscaler_trace::initialize(addon, call_trace_enabled, ngx_source_enabled, ngx_calibration_probe);
    requested.store(false, std::memory_order_release);
    source_requested.store(false, std::memory_order_release);
    observing.store(false, std::memory_order_release);
    observation_generation.fetch_add(1, std::memory_order_acq_rel);
    AcquireSRWLockExclusive(&commands_lock);
    command_tracker.clear();
    content_ledger.clear();
    content_observing = false;
    if (diagnostic_enabled) native_discard::initialize(native_discard_invalidated);
    else native_discard::shutdown();
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
    test_content_requested.store(false, std::memory_order_release);
    test_capture_requested.store(false, std::memory_order_release);
#endif
    command_observation_lost.store(false, std::memory_order_release);
    ReleaseSRWLockExclusive(&commands_lock);
    AcquireSRWLockExclusive(&source_resources_lock);
    source_resources.fill({});
    source_resources_lost.store(false, std::memory_order_release);
    ReleaseSRWLockExclusive(&source_resources_lock);
    addon_module = addon;
    // ReShade may re-register this process-pinned add-on after device recreation.
    // Keep the existing hooks and only start a fresh observation epoch.
    while (!token_metadata.clear()) SwitchToThread();
    release_camera_records();
    AcquireSRWLockExclusive(&presentations_lock);
    for (auto &record : presentations) { if (record.thread) CloseHandle(record.thread); record = {}; }
    lose_presentation();
    lose(loss_diagnostics::reason::lifecycle, __func__, __LINE__, {});
    ++epoch;
    ReleaseSRWLockExclusive(&presentations_lock);
    next_discovery = next_report = 0;
    last_report_activity = UINT64_MAX;
    last_selected = {};
    AcquireSRWLockExclusive(&pcl_lock);
    pcl_available.store(false, std::memory_order_release);
    current_pcl_target.store(nullptr, std::memory_order_release);
    ReleaseSRWLockExclusive(&pcl_lock);
    requested.store(diagnostic_enabled, std::memory_order_release);
    source_requested.store(source_enabled, std::memory_order_release);
    if (hooks_installed) observing.store(diagnostic_enabled || source_enabled, std::memory_order_release);
    if (!diagnostic_enabled)
      message("Sunshine Streamline: diagnostic probe disabled; command/content tracking is off");
  }
  bool enabled() { return requested.load(std::memory_order_acquire); }
  bool source_enabled() { return source_requested.load(std::memory_order_acquire); }
  std::uint64_t depth_observation_revision() { return loss_revision.load(std::memory_order_acquire); }
  bool query_depth_observation_loss(std::uint64_t revision, loss_diagnostics::event &out) {
    return loss_journal.query(revision, out);
  }
  void shutdown() {
    sunshine_upscaler_trace::shutdown();
    native_discard::shutdown();
    requested.store(false, std::memory_order_release);
    source_requested.store(false, std::memory_order_release);
    observing.store(false, std::memory_order_release);
    observation_generation.fetch_add(1, std::memory_order_acq_rel);
    // The shared capture owner retains GPU completion/retirement leases.
    sunshine_game3d::ui_mask::invalidate_all();
    release_camera_records();
  }

  void command_initialized(std::uint64_t native, std::uint64_t device, std::uint64_t cookie) {
    command_event([&](commands::tracker &tracker) { tracker.command_initialized(native, device, cookie); });
  }
  void command_destroyed(std::uint64_t native) {
    command_event([&](commands::tracker &tracker) { tracker.command_destroyed(native); content_ledger.command_destroyed(native); });
  }
  void command_reset(std::uint64_t native, std::uint64_t device, std::uint64_t cookie) {
    command_event([&](commands::tracker &tracker) { tracker.command_reset(native, device, cookie); content_ledger.command_reset(native); });
  }
  void command_closed(std::uint64_t native) {
    command_event([&](commands::tracker &tracker) { tracker.command_closed(native); });
  }
  void queue_initialized(std::uint64_t native, std::uint64_t device, std::uint64_t cookie) {
    command_event([&](commands::tracker &tracker) { tracker.queue_initialized(native, device, cookie); });
  }
  void queue_destroyed(std::uint64_t native) {
    command_event([&](commands::tracker &tracker) { tracker.queue_destroyed(native); });
  }
  void command_executed(std::uint64_t queue, std::uint64_t command) {
    command_event([&](commands::tracker &tracker) { tracker.command_executed(queue, command); });
  }
  void command_secondary_executed(std::uint64_t primary, std::uint64_t secondary) {
    command_event([&](commands::tracker &tracker) { tracker.command_secondary_executed(primary, secondary); });
  }
  commands::recording_marker capture_command_marker(std::uint64_t native) {
    commands::recording_marker out;
    const auto ticket = command_ticket();
    if (!ticket) return out;
    if (!TryAcquireSRWLockExclusive(&commands_lock)) {
      // A missed optional marker does not lose lifecycle identity or destroy the
      // live-object maps. Revoke only evidence that depended on this observation.
      dropped_observation(loss_diagnostics::reason::commands_busy, __func__, __LINE__);
      return out;
    }
    if (current_commands(ticket) && !command_observation_lost.load(std::memory_order_acquire))
      out = command_tracker.mark(native, loss_revision.load(std::memory_order_acquire));
    ReleaseSRWLockExclusive(&commands_lock);
    return out;
  }
  void observe_native_discard_command(std::uint64_t native) { native_discard::observe_command(native); }

  void depth_resource_initialized(std::uint64_t native, std::uint64_t lifetime, std::uint64_t device, bool supported) {
    if (supported) source_resource_event(native, lifetime, device, true);
    command_event([&](commands::tracker &) { content_ledger.resource_initialized({native, lifetime}, device, supported); });
  }
  void depth_resource_destroyed(std::uint64_t native, std::uint64_t lifetime) {
    source_resource_event(native, lifetime, 0, false);
    command_event([&](commands::tracker &) { content_ledger.resource_destroyed({native, lifetime}); });
  }
  void depth_content_write(std::uint64_t command, std::uint64_t native, std::uint64_t lifetime) {
    if (!content_requested()) return;
    command_event([&](commands::tracker &tracker) {
      if (content_observing) content_ledger.write(tracker.mark(command, loss_revision.load(std::memory_order_acquire)), {native, lifetime});
    });
  }
  void depth_content_invalidate(std::uint64_t command, std::uint64_t native, std::uint64_t lifetime) {
    if (!content_requested()) return;
    command_event([&](commands::tracker &tracker) {
      if (content_observing) content_ledger.invalidate(tracker.mark(command, loss_revision.load(std::memory_order_acquire)), {native, lifetime});
    });
  }
  content::copy_snapshot record_depth_copy(std::uint64_t command, content::resource_key source, content::resource_key backup) {
    content::copy_snapshot out;
    command_event([&](commands::tracker &tracker) {
      const auto marker = tracker.mark(command, loss_revision.load(std::memory_order_acquire));
      if (content_observing) out = content_ledger.copy(marker, source, backup, tracker.recording_device(marker));
      else out.recording = marker; // Preserve the earlier command-only contract.
    });
    return out;
  }
  content::copy_snapshot forward_depth_copy(std::uint64_t command, const content::copy_snapshot &preserved,
      content::resource_key destination) {
    content::copy_snapshot out;
    command_event([&](commands::tracker &tracker) {
      const auto marker = tracker.mark(command, loss_revision.load(std::memory_order_acquire));
      if (content_observing) out = content_ledger.forward(marker, preserved, destination, tracker.recording_device(marker));
      else out.recording = marker;
    });
    return out;
  }

  evidence_status query_depth_source(depth_source_snapshot &output, std::uint32_t max_age_ms) {
    output = {};
    const auto fail = [&](evidence_status status) { output.status = status; output.tags = {}; return status; };
    const auto ticket = observation_ticket();
    if (!source_enabled() || !ticket) return fail(evidence_status::inactive);
    auto pin = read_metadata();
    if (!pin) return fail(evidence_status::missing_evaluation);
    const auto loss = loss_revision.load(std::memory_order_acquire);
    const auto now = GetTickCount64();
    const viewport_record *chosen = nullptr;
    unsigned recent_viewports = 0;
    for (const auto &record : pin->records) {
      const auto &value = record.evaluation;
      if (!value.sequence) continue;
      if (!chosen || value.sequence > chosen->evaluation.sequence) chosen = &record;
      if (now >= value.tick && now - value.tick <= max_age_ms) ++recent_viewports;
    }
    auto status = evidence_status::missing_evaluation;
    if (chosen) {
      const auto &value = chosen->evaluation;
      output.epoch = value.epoch; output.sequence = value.sequence; output.tick = value.tick;
      output.loss_revision = value.loss_revision; output.frame = value.frame; output.viewport = value.viewport;
      status = value.status;
      const auto *frame = find_frame(*pin, value.viewport, value.frame);
      if (recent_viewports > 1) status = evidence_status::ambiguous_viewport;
      else if (status == evidence_status::source_associated_evaluation) {
        if (!value.successful_evaluation || !value.frame_correlated || !value.projection.valid()) status = evidence_status::invalid_camera;
        else if (value.loss_revision != loss) status = evidence_status::observation_lost;
        else if (!token_current(*pin, value.frame)) status = evidence_status::untracked_frame;
        else if (!frame || frame->camera_serial != value.camera_sequence) status = evidence_status::stale;
        else if (now < value.tick || now - value.tick > max_age_ms ||
            now < value.camera_tick || now - value.camera_tick > max_age_ms) status = evidence_status::stale;
      }
      if (status == evidence_status::source_associated_evaluation) {
        bool available = false;
        for (unsigned order = 0; order != 2; ++order) {
          const unsigned index = order == 0 ? 1 : 0;
          const auto &tag = value.tags[index];
          if (!tag.present || !tag.value.native_resource) continue;
          if (!tag.supported || match({value.viewport, value.epoch, 0, false}, tag.value) != match_status::same_epoch_only) {
            status = evidence_status::unsupported_depth; break;
          }
          if (tag.scope != tag_scope::evaluation_local) {
            const auto *active = tag.scope == tag_scope::active_global ? &chosen->active_tags[index] :
              frame ? &frame->tags[index] : nullptr;
            if (!active || !active->present || !active->supported ||
                active->value.native_resource != tag.value.native_resource || active->value.lifecycle != tag.value.lifecycle ||
                active->value.resource_width != tag.value.resource_width || active->value.resource_height != tag.value.resource_height ||
                active->resource_lifetime != tag.resource_lifetime || active->resource_device != tag.resource_device ||
                std::memcmp(&active->value.area, &tag.value.area, sizeof(extent)) != 0) {
              status = evidence_status::source_mismatch; break;
            }
          }
          // This identity was bound before the evaluation, not looked up and
          // attached to an old tag after a native pointer was recycled.
          const auto live = source_identity(tag.value.native_resource);
          if (!tag.resource_lifetime || !tag.resource_device ||
              live.lifetime != tag.resource_lifetime || live.device != tag.resource_device) continue;
          output.tags[order] = {tag.value, tag.resource_lifetime, tag.resource_device, true};
          available = true;
        }
        if (status == evidence_status::source_associated_evaluation && !available) status = evidence_status::missing_depth;
      }
    }
    pin.reset();
    if (!current(ticket) || !source_enabled()) return fail(evidence_status::inactive);
    if (loss != loss_revision.load(std::memory_order_acquire) || source_resources_lost.load(std::memory_order_acquire))
      return fail(evidence_status::observation_lost);
    if (status != evidence_status::source_associated_evaluation) return fail(status);
    output.status = status;
    return status;
  }

  evidence_status query_evaluation(const selected_depth &selected, evaluation_snapshot &output, std::uint32_t max_age_ms) {
    output = {};
    const auto ticket = observation_ticket();
    const auto fail = [&](evidence_status status) { output.status = status; return status; };
    if (!ticket || !enabled()) return fail(evidence_status::inactive);
    if (!selected.ready || !selected.resource || !selected.width || !selected.height ||
        !selected.active_width || !selected.active_height ||
        static_cast<std::uint64_t>(selected.x) + selected.active_width > selected.width ||
        static_cast<std::uint64_t>(selected.y) + selected.active_height > selected.height)
      return fail(evidence_status::depth_not_ready);
    auto pin = read_metadata();
    if (!pin) return fail(evidence_status::missing_evaluation);
    const auto loss = loss_revision.load(std::memory_order_acquire);
    const auto now = GetTickCount64();
    evaluation_snapshot newest;
    bool found = false, ambiguous = false;
    for (const auto &record : pin->records) {
      const auto &candidate = record.evaluation;
      if (!candidate.sequence) continue;
      if (candidate.sequence > newest.sequence) newest = candidate;
      unsigned matching = static_cast<unsigned>(candidate.tags.size());
      bool source_seen = false;
      auto status = candidate.status;
      const auto *frame_state = find_frame(*pin, candidate.viewport, candidate.frame);
      for (unsigned i = 0; i != 2; ++i) {
        const auto &tag = candidate.tags[i];
        if (!tag.present || tag.value.native_resource != selected.resource) continue;
        source_seen = true;
        const auto paired = match({candidate.viewport, candidate.epoch, 0, false}, tag.value);
        if (!tag.supported || paired != match_status::same_epoch_only) {
          if (candidate.status == evidence_status::source_associated_evaluation) status = evidence_status::unsupported_depth;
          continue;
        }
        const auto &area = tag.value.area;
        const bool whole = !area.top && !area.left && !area.width && !area.height;
        if ((tag.value.resource_width && tag.value.resource_width != selected.width) ||
            (tag.value.resource_height && tag.value.resource_height != selected.height) ||
            (whole ? selected.x || selected.y || selected.active_width != selected.width || selected.active_height != selected.height :
              area.left != selected.x || area.top != selected.y || area.width != selected.active_width || area.height != selected.active_height)) {
          if (candidate.status == evidence_status::source_associated_evaluation) status = evidence_status::extent_mismatch;
          continue;
        }
        // A global tag changed after evaluation entry. The old immutable tuple
        // remains diagnostic evidence but is no longer an applicable binding.
        if (tag.scope != tag_scope::evaluation_local) {
          const auto *active = tag.scope == tag_scope::active_global ? &record.active_tags[i] :
            frame_state ? &frame_state->tags[i] : nullptr;
          if (!active || !active->present || !active->supported ||
              active->value.native_resource != tag.value.native_resource || active->value.lifecycle != tag.value.lifecycle ||
              active->value.resource_width != tag.value.resource_width || active->value.resource_height != tag.value.resource_height ||
              std::memcmp(&active->value.area, &tag.value.area, sizeof(extent)) != 0) {
            if (candidate.status == evidence_status::source_associated_evaluation) status = evidence_status::source_mismatch;
            continue;
          }
        }
        matching = i;
        status = candidate.status;
        break;
      }
      if (!source_seen) continue;
      if (found) { ambiguous = true; continue; }
      found = true;
      output = candidate;
      output.selection = selected;
      if (matching != candidate.tags.size()) output.matched_tag = matching;
      if (status == evidence_status::source_associated_evaluation) {
        if (!candidate.successful_evaluation || !candidate.frame_correlated || !candidate.projection.valid()) status = evidence_status::invalid_camera;
        else if (candidate.loss_revision != loss) status = evidence_status::observation_lost;
        else if (!token_current(*pin, candidate.frame)) status = evidence_status::untracked_frame;
        else if (!frame_state || frame_state->camera_serial != candidate.camera_sequence) status = evidence_status::stale;
        else if (now < candidate.tick || now - candidate.tick > max_age_ms ||
            now < candidate.camera_tick || now - candidate.camera_tick > max_age_ms) status = evidence_status::stale;
      }
      output.status = status;
    }
    pin.reset();
    if (!current(ticket)) return fail(evidence_status::inactive);
    if (loss != loss_revision.load(std::memory_order_acquire)) return fail(evidence_status::observation_lost);
    if (ambiguous) return fail(evidence_status::ambiguous_viewport);
    if (!found) {
      output = newest;
      output.selection = selected;
      return fail(!newest.sequence ? evidence_status::missing_evaluation :
        newest.status == evidence_status::source_associated_evaluation ? evidence_status::source_mismatch : newest.status);
    }
    if (output.status == evidence_status::source_associated_evaluation) associate_evidence(output, selected);
    if (!current(ticket)) return fail(evidence_status::inactive);
    if (loss != loss_revision.load(std::memory_order_acquire)) return fail(evidence_status::observation_lost);
    return output.status;
  }

  bool query_frame_generation(std::uint32_t viewport, frame_generation_snapshot &output,
      frame_generation_query_status *status) {
    output = {};
    if (status) *status = frame_generation_query_status::unavailable;
    const auto ticket = observation_ticket();
    if (!ticket || !fg_available.load(std::memory_order_acquire) || !current_fg_target.load(std::memory_order_acquire)) return false;
    auto pin = read_metadata();
    if (!pin) {
      if (status && metadata.has_publication()) *status = frame_generation_query_status::busy;
      return false;
    }
    bool found = false, ambiguous = false;
    const auto options_loss = fg_loss_revision.load(std::memory_order_acquire);
    for (const auto &record : pin->records) {
      const auto &state = record.frame_generation;
      if (!record.used || !state.sequence || (viewport != UINT32_MAX && record.viewport != viewport)) continue;
      if (!state.known || state.epoch != pin->epoch || state.loss_revision != options_loss) { ambiguous = true; continue; }
      if (found) { ambiguous = true; continue; }
      output = state; found = true;
    }
    pin.reset();
    if (!found || ambiguous || !current(ticket) || options_loss != fg_loss_revision.load(std::memory_order_acquire)) {
      if (status && ambiguous) *status = frame_generation_query_status::ambiguous;
      output = {}; return false;
    }
    if (status) *status = frame_generation_query_status::observed;
    return true;
  }

  presentation_status query_current_presentation(presentation_snapshot &output, std::uint32_t max_age_ms) {
    output = {};
    const auto ticket = observation_ticket();
    const auto fail = [&](presentation_status status) { output.status = status; output.explicit_bracket = false; return status; };
    if (!ticket) return fail(presentation_status::inactive);
    if (installed_abi == abi::v2_7_30 && (!pcl_available.load(std::memory_order_acquire) || !current_pcl_target.load(std::memory_order_acquire)))
      return fail(presentation_status::missing_marker_function);
    if (!TryAcquireSRWLockExclusive(&presentations_lock)) return fail(presentation_status::busy);
    const auto loss = loss_revision.load(std::memory_order_acquire);
    const auto marker_loss = presentation_loss.load(std::memory_order_acquire);
    const auto *record = current(ticket) ? presentation_thread(false) : nullptr;
    if (record) {
      output = record->value;
      if ((output.status == presentation_status::open_bracket || output.status == presentation_status::pending_marker) &&
          (output.epoch != epoch || output.loss_revision != loss || output.marker_loss_revision != marker_loss))
        fail(presentation_status::observation_lost);
      else if (output.status == presentation_status::open_bracket && !presentation_token_current(output.frame)) fail(presentation_status::untracked_frame);
      else if (output.status == presentation_status::open_bracket) {
        const auto now = GetTickCount64();
        if (now < output.start_tick || now - output.start_tick > max_age_ms) fail(presentation_status::stale);
      }
    }
    ReleaseSRWLockExclusive(&presentations_lock);
    if (!current(ticket)) return fail(presentation_status::inactive);
    if (loss != loss_revision.load(std::memory_order_acquire) || marker_loss != presentation_loss.load(std::memory_order_acquire))
      return fail(presentation_status::observation_lost);
    return output.status;
  }
  const char *name(presentation_status value) {
    switch (value) {
      case presentation_status::inactive: return "inactive";
      case presentation_status::missing_marker_function: return "missing-pcl-marker-function";
      case presentation_status::missing_bracket: return "missing-thread-present-bracket";
      case presentation_status::pending_marker: return "present-marker-pending";
      case presentation_status::open_bracket: return "explicit-thread-present-bracket";
      case presentation_status::ended_bracket: return "present-bracket-ended";
      case presentation_status::failed_marker: return "present-marker-failed";
      case presentation_status::untracked_frame: return "untracked-present-frame";
      case presentation_status::out_of_order: return "present-markers-out-of-order";
      case presentation_status::repeated_frame: return "repeated-frame-presentation";
      case presentation_status::stale: return "stale-present-bracket";
      case presentation_status::observation_lost: return "lost-present-observation";
      case presentation_status::busy: return "present-snapshot-busy";
    }
    return "unknown";
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

  void poll(const selected_depth &selected) {
    const bool api_requested = middleware_requested() || sunshine_upscaler_trace::enabled() ||
      sunshine_upscaler_trace::capture_enabled();
    if (!api_requested && !depth_capture::active()) return;
    if (!TryAcquireSRWLockExclusive(&poll_lock)) return;
    struct release_poll { ~release_poll() { ReleaseSRWLockExclusive(&poll_lock); } } release;
    const auto now = GetTickCount64();
    if (api_requested && now >= next_discovery) {
      next_discovery = now + 1000;
      if (!hooks_installed && !permanently_rejected) discover();
      if (hooks_installed) discover_v1_private_state();
      if (hooks_installed) discover_fg_options();
    }
    // Call routing is independent of per-draw diagnostics and source selection.
    // A zero count is only useful together with actual hook coverage.
    sunshine_upscaler_trace::set_streamline_coverage(hooks_installed);
    sunshine_upscaler_trace::poll();
    // Camera availability is useful even when this game sends its upscaling
    // through NGX. This is observation only, never a cross-API frame match.
    if (sunshine_upscaler_trace::capture_enabled() && hooks_installed) {
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
        std::array<depth_tag_sample, 2> depth_samples{};
        for (const auto &record : camera_pin->records) if (record.used && record.has_camera) {
          ++cameras;
          const auto checked = validate(record.camera);
          valid_recent += record.camera_tick <= now && now - record.camera_tick <= 250 &&
            record.decoded == decode_status::ok && checked.valid();
          if (record.decoded == decode_status::ok && record.camera_tick >= camera_sample_tick) {
            camera_sample = record.camera;
            camera_sample_key = record.key;
            camera_sample_validation = checked;
            camera_sample_tick = record.camera_tick;
            depth_samples = {};
            for (const auto *group : {&record.active_tags, &record.global_tags, &record.local_tags}) {
              for (const auto &tag : *group) {
                if (!tag.present || (tag.value.type != 0 && tag.value.type != 48)) continue;
                auto &sample = depth_samples[tag.value.type == 0 ? 0 : 1];
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
          char availability[320]{};
          std::snprintf(availability, sizeof(availability),
            "Sunshine Streamline camera availability: constants_calls=%llu cameras=%u recent_valid_projection=%u; NGX_frame_match=unproven; availability does not authorize cross-API scale",
            static_cast<unsigned long long>(calls), cameras, valid_recent);
          message(availability);
          if (camera_sample_tick) {
            char detail[1400]{};
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
            char tag_details[2][480]{};
            for (unsigned i = 0; i < depth_samples.size(); ++i) {
              const auto &sample = depth_samples[i];
              const auto &tag = sample.value;
              std::snprintf(tag_details[i], sizeof(tag_details[i]),
                "type=%u present=%u supported=%u scope=%u native=0x%llx declared_size=%ux%u area_xywh=%u,%u,%u,%u lifetime=%u age_ms=%llu tag_viewport=%u frame=%llu explicit_frame=%u",
                i == 0 ? 0u : 48u, sample.present ? 1u : 0u,
                sample.supported ? 1u : 0u, static_cast<unsigned>(sample.scope),
                static_cast<unsigned long long>(tag.native_resource), tag.resource_width, tag.resource_height,
                tag.area.left, tag.area.top, tag.area.width, tag.area.height, tag.lifecycle,
                static_cast<unsigned long long>(sample.present && sample.tick <= now ? now - sample.tick : UINT64_MAX),
                tag.viewport, static_cast<unsigned long long>(tag.frame_key), tag.explicit_frame ? 1u : 0u);
            }
            std::snprintf(detail, sizeof(detail),
              "Sunshine Streamline depth tag sample: camera_viewport=%u depth={%s} highres_depth={%s}; observation only, NGX_resource_overlap=unproven",
              camera_sample_key.viewport, tag_details[0], tag_details[1]);
            message(detail);
          }
          camera_report_count = calls;
        }
      }
    }
    if (content_requested()) install_discard_observations();
    // The common copy owner also serves Generic depth without an upscaler API.
    // Middleware presence and selection preferences cannot gate its lifecycle.
    if (depth_capture::active()) {
      if (!minhook_initialized) {
        const auto initialized = MH_Initialize();
        minhook_initialized = initialized == MH_OK || initialized == MH_ERROR_ALREADY_INITIALIZED;
      }
      if (minhook_initialized) depth_capture::poll();
    }
    if (middleware_requested() && observing.load(std::memory_order_acquire)) {
      install_pcl_hooks();
      install_fg_hooks();
    }
    if (!enabled() || !hooks_installed || !observing.load(std::memory_order_acquire) || now < next_report) return;
    next_report = now + 5000;
    auto pin = read_metadata();
    if (!pin) return;
    const auto &copy = pin->records;
    const auto copied_revision = pin->revision;
    const auto snapshot_now = GetTickCount64();
    // Stable/no-call states are reported only once. Active games have bounded
    // five-second numeric snapshots, never per-frame text or file I/O in hooks.
    const auto activity = copied_revision + constant_calls.load() + tag_calls.load() + evaluation_calls.load();
    const bool selection_changed = selected.resource != last_selected.resource || selected.source_id != last_selected.source_id ||
      selected.layout_epoch != last_selected.layout_epoch || selected.ready != last_selected.ready ||
      selected.width != last_selected.width || selected.height != last_selected.height ||
      selected.x != last_selected.x || selected.y != last_selected.y ||
      selected.active_width != last_selected.active_width || selected.active_height != last_selected.active_height;
    if (last_report_activity == activity && !selection_changed) return;
    last_report_activity = activity;
    last_selected = selected;
    char text[3072];
    std::snprintf(text, sizeof(text),
      "Sunshine Streamline: epoch=%llu calls constants=%llu tags=%llu evaluations=%llu dropped=%llu invalid=%llu selected=0x%llx source=%llu layout=%llu present=%llu ready=%d dimensions=%ux%u region=%u,%u,%u,%u diagnostic_only=1",
      static_cast<unsigned long long>(epoch), static_cast<unsigned long long>(constant_calls.load()),
      static_cast<unsigned long long>(tag_calls.load()), static_cast<unsigned long long>(evaluation_calls.load()),
      static_cast<unsigned long long>(dropped.load()), static_cast<unsigned long long>(invalid.load()),
      static_cast<unsigned long long>(selected.resource), static_cast<unsigned long long>(selected.source_id),
      static_cast<unsigned long long>(selected.layout_epoch), static_cast<unsigned long long>(selected.frame_index),
      selected.ready ? 1 : 0, selected.width, selected.height, selected.x, selected.y, selected.active_width, selected.active_height);
    message(text);
    evaluation_snapshot evidence;
    query_evaluation(selected, evidence);
    std::snprintf(text, sizeof(text),
      "Sunshine Streamline evaluation: status=%s viewport=%u sequence=%llu success=%d constants_frame_correlated=%d frame_kind=%u token_generation=%llu numeric_frame=%llu explicit_numeric=%d commands=0x%llx camera_age_ms=%llu reshade_frame_correlated=0 units=unknown diagnostic_only=1",
      name(evidence.status), evidence.viewport, static_cast<unsigned long long>(evidence.sequence),
      evidence.successful_evaluation ? 1 : 0, evidence.frame_correlated ? 1 : 0, static_cast<unsigned>(evidence.frame.kind),
      static_cast<unsigned long long>(evidence.frame.generation), static_cast<unsigned long long>(evidence.frame.numeric),
      evidence.frame.has_numeric ? 1 : 0, static_cast<unsigned long long>(evidence.command_buffer),
      static_cast<unsigned long long>(evidence.camera_tick ? GetTickCount64() - evidence.camera_tick : UINT64_MAX));
    message(text);
    // Keep projection and association in one evaluated tuple. The latest viewport
    // constants printed below can belong to a different frame and must not be joined by time.
    const auto &camera = evidence.camera;
    const auto &tag = evidence.tags[evidence.matched_tag < evidence.tags.size() ? evidence.matched_tag : 0];
    std::snprintf(text, sizeof(text),
      "Sunshine Streamline evaluated camera: status=%s viewport=%u sequence=%llu frame_kind=%u numeric_frame=%llu explicit_numeric=%d token_generation=%llu validation=%s A=%.17g B=%.17g near=%.9g far=%.9g fov=%.9g aspect=%.9g reset=%u selected=0x%llx source=%llu layout=%llu reshade_present=%llu dimensions=%ux%u tag_present=%d tag_type=%u tag_resource=0x%llx tag_extent=%u,%u,%u,%u units=game-units metric_scale_known=0 diagnostic_only=1",
      name(evidence.status), evidence.viewport, static_cast<unsigned long long>(evidence.sequence),
      static_cast<unsigned>(evidence.frame.kind), static_cast<unsigned long long>(evidence.frame.numeric),
      evidence.frame.has_numeric ? 1 : 0, static_cast<unsigned long long>(evidence.frame.generation),
      name(evidence.projection.status), evidence.projection.depth_offset, evidence.projection.depth_scale,
      camera.near_plane, camera.far_plane, camera.fov, camera.aspect, camera.reset,
      static_cast<unsigned long long>(selected.resource), static_cast<unsigned long long>(selected.source_id),
      static_cast<unsigned long long>(selected.layout_epoch), static_cast<unsigned long long>(selected.frame_index),
      selected.width, selected.height, tag.present ? 1 : 0, tag.value.type,
      static_cast<unsigned long long>(tag.value.native_resource), tag.value.area.left, tag.value.area.top,
      tag.value.area.width, tag.value.area.height);
    message(text);
    std::snprintf(text, sizeof(text),
      "Sunshine Streamline command association: status=%s stable_recording=%d queue=0x%llx queue_generation=%llu copy_submission=%llu evaluation_submission=%llu order=%u same_recording=%d depth_content_registered=0 final_color_registered=0 gpu_completion_proven=0",
      commands::name(evidence.command_association.state), evidence.recording_stable ? 1 : 0,
      static_cast<unsigned long long>(evidence.command_association.queue),
      static_cast<unsigned long long>(evidence.command_association.queue_generation),
      static_cast<unsigned long long>(evidence.command_association.copy_submission),
      static_cast<unsigned long long>(evidence.command_association.evaluation_submission),
      static_cast<unsigned>(evidence.command_association.ordering), evidence.command_association.same_recording ? 1 : 0);
    message(text);
    std::snprintf(text, sizeof(text),
      "Sunshine Streamline content association: status=%s observed_content_match=%d coverage_complete=0 final_color_registered=0 diagnostic_only=1",
      content::name(evidence.content_association.state), evidence.content_association.matched() ? 1 : 0);
    message(text);
    const auto discard = native_discard::counts();
    std::snprintf(text, sizeof(text),
      "Sunshine native discard observation: targets=%llu installed=%llu calls=%llu observed=%llu unreadable=%llu dropped=%llu rejected=%llu complete_mutation_coverage=0",
      static_cast<unsigned long long>(discard.targets), static_cast<unsigned long long>(discard.installed),
      static_cast<unsigned long long>(discard.calls), static_cast<unsigned long long>(discard.observed),
      static_cast<unsigned long long>(discard.unreadable), static_cast<unsigned long long>(discard.dropped),
      static_cast<unsigned long long>(discard.rejected));
    message(text);
    presentation_snapshot present;
    query_current_presentation(present);
    const bool same_present_frame = present.explicit_bracket && evidence.successful_evaluation &&
      evidence.frame_correlated && same_frame(present.frame, evidence.frame);
    std::snprintf(text, sizeof(text),
      "Sunshine presentation trace: status=%s thread=%u thread_generation=%llu frame_kind=%u numeric_frame=%llu token_generation=%llu start=%llu end=%llu explicit_bracket=%d evaluation_frame_match=%d pcl_getters=%llu pcl_markers=%llu dropped=%llu effects_runtime=0x%llx effects_resource=0x%llx effects_size=%ux%u effects_format=%u final_color_registered=0",
      name(present.status), present.thread_id, static_cast<unsigned long long>(present.thread_generation), static_cast<unsigned>(present.frame.kind),
      static_cast<unsigned long long>(present.frame.numeric), static_cast<unsigned long long>(present.frame.generation),
      static_cast<unsigned long long>(present.start_sequence), static_cast<unsigned long long>(present.end_sequence), present.explicit_bracket ? 1 : 0, same_present_frame ? 1 : 0,
      static_cast<unsigned long long>(pcl_getter_calls.load()), static_cast<unsigned long long>(pcl_calls.load()),
      static_cast<unsigned long long>(presentation_dropped.load()), static_cast<unsigned long long>(selected.effects_input.runtime),
      static_cast<unsigned long long>(selected.effects_input.resource), selected.effects_input.width, selected.effects_input.height, selected.effects_input.format);
    message(text);
    for (const auto &color : evidence.colors) if (color.present) {
      const bool identity = selected.effects_input.ready && color.identity_available && color.value.native_resource == selected.effects_input.resource;
      const bool dimensions = color.value.resource_width && color.value.resource_height && selected.effects_input.ready &&
        color.value.resource_width == selected.effects_input.width && color.value.resource_height == selected.effects_input.height;
      std::snprintf(text, sizeof(text),
        "Sunshine Streamline color: type=%u scope=%u native=0x%llx size=%ux%u extent=%u,%u,%u,%u lifecycle=%u supported=%d identity_available=%d effects_identity_match=%d declared_size_match=%d final_color_registered=0",
        color.value.type, static_cast<unsigned>(color.scope), static_cast<unsigned long long>(color.value.native_resource),
        color.value.resource_width, color.value.resource_height, color.value.area.left, color.value.area.top, color.value.area.width, color.value.area.height,
        color.value.lifecycle, color.supported ? 1 : 0, color.identity_available ? 1 : 0, identity ? 1 : 0, dimensions ? 1 : 0);
      message(text);
    }
    for (const auto &record : copy) {
      if (!record.used) continue;
      const auto checked = validate(record.camera);
      std::snprintf(text, sizeof(text),
        "Sunshine Streamline camera: viewport=%u age_ms=%llu decode=%s validation=%s numeric_frame=%llu explicit_frame=%d token=0x%llx feature=%u evaluate_age_ms=%llu near=%.9g far=%.9g fov=%.9g aspect=%.9g inverted=%u orthographic=%u reset=%u inactive=%u A=%.17g B=%.17g inverse_error=%.9g matrix_near=%.9g matrix_far=%.9g matrix_fov=%.9g matrix_aspect=%.9g right_handed=%d infinite_far=%d units=unknown",
        record.viewport, static_cast<unsigned long long>(record.has_camera ? snapshot_now - record.camera_tick : UINT64_MAX),
        name(record.decoded), record.has_camera ? name(checked.status) : "missing-constants",
        static_cast<unsigned long long>(record.key.frame_key), record.key.explicit_frame ? 1 : 0,
        static_cast<unsigned long long>(record.camera_token), record.feature,
        static_cast<unsigned long long>(record.evaluation_tick ? snapshot_now - record.evaluation_tick : UINT64_MAX),
        record.camera.near_plane, record.camera.far_plane, record.camera.fov, record.camera.aspect,
        record.camera.depth_inverted, record.camera.orthographic,
        record.camera.reset, record.camera.not_rendering_game_frames, checked.depth_offset, checked.depth_scale,
        checked.inverse_error, checked.near_from_matrix, checked.far_from_matrix, checked.fov_from_matrix,
        checked.aspect_from_matrix, checked.right_handed ? 1 : 0, checked.infinite_far ? 1 : 0);
      message(text);
      const float *p = &record.camera.projection.m[0][0], *q = &record.camera.inverse_projection.m[0][0];
      std::snprintf(text, sizeof(text),
        "Sunshine Streamline matrices: viewport=%u viewToClip=[%.9g,%.9g,%.9g,%.9g;%.9g,%.9g,%.9g,%.9g;%.9g,%.9g,%.9g,%.9g;%.9g,%.9g,%.9g,%.9g] clipToView=[%.9g,%.9g,%.9g,%.9g;%.9g,%.9g,%.9g,%.9g;%.9g,%.9g,%.9g,%.9g;%.9g,%.9g,%.9g,%.9g]",
        record.viewport, p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15],
        q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15]);
      message(text);
      for (const auto *group : {&record.global_tags, &record.local_tags}) for (const auto &value : *group) {
        const auto *tag = &value;
        if (!tag->present || tag_index(tag->value.type) >= 3) continue;
        const auto matching = match(record.key, tag->value);
        std::snprintf(text, sizeof(text),
          "Sunshine Streamline depth: viewport=%u scope=%s age_ms=%llu native=0x%llx type=%u lifecycle=%u extent=%u,%u,%u,%u resource_size=%ux%u supported=%d pairing=%s token=0x%llx same_token_address=%d exact_frame=0 selected_identity_match=%d freshness_is_not_frame_proof=1",
          record.viewport, source_name(tag->source), static_cast<unsigned long long>(snapshot_now - tag->tick),
          static_cast<unsigned long long>(tag->value.native_resource), tag->value.type, tag->value.lifecycle,
          tag->value.area.left, tag->value.area.top, tag->value.area.width, tag->value.area.height,
          tag->value.resource_width, tag->value.resource_height, tag->supported ? 1 : 0,
          !tag->supported ? "unsupported-tag-metadata" : record.has_camera ? name(matching) : "missing-constants", static_cast<unsigned long long>(tag->token),
          record.camera_token && record.camera_token == tag->token ? 1 : 0,
          selected.ready && selected.resource != 0 && selected.resource == tag->value.native_resource ? 1 : 0);
        message(text);
      }
    }
  }

#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
  static std::uint64_t test_capture(std::uint64_t command,
      std::uint64_t resource, std::uint32_t native_state, const camera_data *camera,
      std::uint64_t sequence, bool successful, bool version_one, bool defer_finish = false) {
    // Exercise the real capture boundary using actual native resources/CLs.
    // This seam never supplies readiness, selected depth, fences or uniforms.
    test_capture_requested.store(true, std::memory_order_release);
    depth_capture::observe_provider(command);
    camera_data copied;
    if (!camera || !read(camera, copied) || !sequence) return 0;
    depth_capture::input value;
    value.epoch = 0x53554e534c544553ull; value.sequence = sequence; value.tick = GetTickCount64();
    value.observation_revision = depth_observation_revision();
    const auto projection = validate(copied);
    value.projection = {projection.depth_offset, projection.depth_scale,
      projection.valid() && copied.reset == 0, projection.reversed};
    value.native_state = native_state;
    value.proof = version_one && native_state == 0 ? sunshine_scene_depth::state_proof::observed_nonzero :
      sunshine_scene_depth::state_proof::declared;
    value.resource.native = resource;
    value.valid_until = sunshine_scene_depth::lifetime::until_present;
    value.source = depth_capture::retain_source(resource);
    value.source_present_generation = depth_capture::source_present_generation(value.source);
    if (value.source) {
      const auto state = v1_resource_state::resolve(native_state, version_one,
        reinterpret_cast<IUnknown *>(resource));
      value.native_state = state.native;
      value.proof = state.proof;
    }
    depth_capture::begin_evaluation(value.epoch, sequence, 0);
    const auto ticket = depth_capture::nominate(command, value);
    if (ticket && !defer_finish) depth_capture::finish(ticket, successful);
    return ticket;
  }
  extern "C" __declspec(dllexport) std::uint64_t SunshineStreamlineTestCapture(std::uint64_t command,
      std::uint64_t resource, std::uint32_t native_state, const camera_data *camera, std::uint64_t sequence, bool successful) {
    return test_capture(command, resource, native_state, camera, sequence, successful, false);
  }
  extern "C" __declspec(dllexport) std::uint64_t SunshineStreamlineTestCaptureV1(std::uint64_t command,
      std::uint64_t resource, std::uint32_t native_state, const camera_data *camera, std::uint64_t sequence, bool successful) {
    return test_capture(command, resource, native_state, camera, sequence, successful, true);
  }
  extern "C" __declspec(dllexport) std::uint64_t SunshineStreamlineTestBeginCaptureV1(std::uint64_t command,
      std::uint64_t resource, std::uint32_t native_state, const camera_data *camera, std::uint64_t sequence, bool) {
    return test_capture(command, resource, native_state, camera, sequence, false, true, true);
  }
  extern "C" __declspec(dllexport) void SunshineStreamlineTestFinishCapture(std::uint64_t ticket, bool successful) {
    if (ticket) depth_capture::finish(ticket, successful);
  }
  extern "C" __declspec(dllexport) BOOL SunshinePresentationTestQuery(presentation_snapshot *out) {
    if (!out) return FALSE;
    query_current_presentation(*out);
    return TRUE;
  }
  extern "C" __declspec(dllexport) BOOL SunshineDiscardTestInstallPending() {
    if (!TryAcquireSRWLockExclusive(&poll_lock)) return FALSE;
    const bool result = install_discard_observations();
    ReleaseSRWLockExclusive(&poll_lock);
    return result ? TRUE : FALSE;
  }
  extern "C" __declspec(dllexport) BOOL SunshineDiscardTestCounts(native_discard::counters *out) {
    if (!out) return FALSE;
    *out = native_discard::counts();
    return TRUE;
  }
  extern "C" __declspec(dllexport) void SunshineContentTestEnable(BOOL enabled) {
    test_content_requested.store(enabled != FALSE, std::memory_order_release);
    command_event([](commands::tracker &) {}); // Serialize epoch transition.
  }
  extern "C" __declspec(dllexport) void SunshineCommandTestLoseLifecycle() {
    command_observation_lost.store(true, std::memory_order_release);
    dropped_observation(loss_diagnostics::reason::test_injected, __func__, __LINE__, {});
  }
  extern "C" __declspec(dllexport) BOOL SunshineCommandTestCapture(std::uint64_t command, commands::recording_marker *out) {
    if (!out) return FALSE;
    *out = capture_command_marker(command);
    return *out ? TRUE : FALSE;
  }
  extern "C" __declspec(dllexport) BOOL SunshineCommandTestAssociate(const commands::recording_marker *evaluation,
      const commands::recording_marker *copy, std::uint64_t queue, commands::association *out) {
    if (!evaluation || !copy || !out) return FALSE;
    *out = associate_commands(*evaluation, *copy, queue);
    return TRUE;
  }
  extern "C" __declspec(dllexport) BOOL SunshineContentTestUse(std::uint64_t command,
      std::uint64_t source, std::uint64_t lifetime, content::use_snapshot *out) {
    if (!out) return FALSE;
    *out = {};
    command_event([&](commands::tracker &tracker) {
      const auto marker = tracker.mark(command, loss_revision.load(std::memory_order_acquire));
      if (content_observing) *out = content_ledger.use(marker, source, tracker.recording_device(marker));
      if (out->valid() && (!lifetime || out->resource.lifetime != lifetime)) out->state = content::status::lifetime_mismatch;
    });
    return TRUE;
  }
  extern "C" __declspec(dllexport) BOOL SunshineContentTestAssociate(const content::use_snapshot *use,
      const content::copy_snapshot *copy, std::uint64_t backup, std::uint64_t lifetime, content::association *out) {
    if (!use || !copy || !out) return FALSE;
    *out = associate_content(*use, *copy, use->resource, {backup, lifetime});
    return TRUE;
  }
#endif

#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
  namespace testing {
    bool normalized_source(std::uint32_t viewport, unsigned index, sunshine_scene_depth::frame &out) {
      out = {};
      evaluation_snapshot snapshot;
      if (!latest_snapshot(viewport, snapshot) || index >= snapshot.tags.size() ||
          !snapshot.successful_evaluation || !source_path_selected(snapshot) || !direct_source_admissible(snapshot) ||
          !direct_tag_admissible(snapshot, snapshot.tags[index])) return false;
      out = normalize_depth_frame(snapshot, snapshot.tags[index], snapshot.frame.kind == frame_identity_kind::v1_numeric);
      return true;
    }
    bool latest_snapshot(std::uint32_t viewport, evaluation_snapshot &out) {
      auto pin = read_metadata();
      if (!pin) return false;
      bool found = false;
      for (const auto &record : pin->records) if (record.evaluation.sequence && record.evaluation.viewport == viewport) {
        out = record.evaluation; found = true; break;
      }
      return found;
    }
    void lock_source_resources() { AcquireSRWLockExclusive(&source_resources_lock); }
    void unlock_source_resources() { ReleaseSRWLockExclusive(&source_resources_lock); }
    unsigned waiting_source_resources() { return source_lifecycle_waiters.load(std::memory_order_acquire); }
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
    void arm_entry_policy() { entry_policy_armed = true; entry_policy_observed = false; }
    bool entry_policy(bool &selected, bool &admissible) {
      selected = entry_path_selected; admissible = entry_source_admitted;
      return entry_policy_observed;
    }
    bool install_presentation_hooks() { return install_pcl_hooks(); }
    bool discover_frame_generation_hooks() { discover_fg_options(); return install_fg_hooks(); }
    bool install(testing::abi version, const testing::targets &functions) {
      if (!middleware_requested() && !sunshine_upscaler_trace::enabled()) return false;
      const auto native_version = version == testing::abi::v1_1_1 ? sunshine_streamline::abi::v1_1_1 :
        version == testing::abi::v2_7_30 ? sunshine_streamline::abi::v2_7_30 : sunshine_streamline::abi::unsupported;
      const bool success = install_hooks(native_version, {functions.constants, functions.tag, functions.tag_for_frame, functions.evaluate, functions.new_frame_token, functions.get_feature_function});
      if (success) observing.store(middleware_requested(), std::memory_order_release);
      sunshine_upscaler_trace::set_streamline_coverage(success);
      return success;
    }
    counters counts() { return {constant_calls.load(), tag_calls.load(), evaluation_calls.load(), dropped.load(), invalid.load()}; }
    void lose_observation() { dropped_observation(loss_diagnostics::reason::test_injected, __func__, __LINE__, {}); }
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
      for (unsigned i = 0; i != pcl_target_count; ++i) {
        if (pcl_targets[i].owner) FreeLibrary(pcl_targets[i].owner);
        pcl_targets[i].entry = nullptr; pcl_targets[i].owner = nullptr;
        pcl_targets[i].original = nullptr; pcl_targets[i].state = 0;
      }
      pcl_target_count = 0; current_pcl_target = nullptr; pcl_available = false;
      for (unsigned i = 0; i != fg_target_count; ++i) {
        if (fg_targets[i].owner) FreeLibrary(fg_targets[i].owner);
        fg_targets[i].entry = nullptr; fg_targets[i].owner = nullptr;
        fg_targets[i].original = nullptr; fg_targets[i].state = 0;
      }
      fg_target_count = 0; current_fg_target = nullptr; fg_available = false;
      fg_loss_revision = 0;
      pcl_calls = pcl_getter_calls = presentation_dropped = 0;
      original_feature_function_v2 = nullptr;
      hooks_installed = minhook_initialized = false;
      v1_private_state_ready.store(false, std::memory_order_release);
      v1_private_state_checked = false;
      installed_abi = sunshine_streamline::abi::unsupported;
      installed_targets = {};
      constant_calls = tag_calls = evaluation_calls = dropped = invalid = 0;
      initialize(nullptr);
    }
  }
#endif
}
