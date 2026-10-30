// SPDX-License-Identifier: GPL-3.0-only
#include "streamline_depth_capture.h"
#include "capture_state_policy.h"
#include "streamline_native_observer.h"
#include "native_command_storage.h"
#include "native_resource_identity.h"
#include "depth_view_format.h"
#include "freshness.h"
#include "game3d_slow_step.h"
#include "addon_lifetime.h"
#include "streamline_depth_capture_test_access.h"
#include "../../src/game3d_debug_formats.h"
#include <d3d12.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <memory>
#include <optional>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace sunshine_streamline::depth_capture {
  SUNSHINE_CAPTURE_INTERNAL {
    template<class T> bool query_native(std::uint64_t native, REFIID iid, com_ptr<T> &out) {
      // Boundary inputs are live COM objects, but their incoming interface may
      // be only IUnknown/base. Never read larger vtables before exact QI.
      return native && SUCCEEDED(reinterpret_cast<IUnknown *>(native)->QueryInterface(iid,
        reinterpret_cast<void **>(out.put()))) && out.p;
    }
    std::atomic<std::uint64_t> serial{1};
    std::atomic<status> reason{status::inactive};
    std::array<std::atomic<status>, provider_count> attempt_status{status::unavailable, status::unavailable};
    std::atomic<bool> requested{};
    std::mutex mutex;
    // Every D3D12 Present the add-on saw (present()), counted before that
    // Present's own selection. An evaluation's view and its snapshot are
    // stamped with present_ordinal() at their call: the ordinal of the
    // Present that presents the frame being recorded then.
    std::atomic<std::uint64_t> presents{};
    // A stamp is the frame the Present being decided presents when no other
    // Present came since its call (frame_presenting's first rule, the only
    // one without frame generation). Such a view is active and such a
    // snapshot current however long the frame took after its call (a hitch
    // between an SR evaluation and its Present), as they are the depth of
    // the frame that Present shows: the source age bounds holds over later
    // Presents, never a frame's own depth at its own Present (10-27; until
    // then that Present was mono, inactive_views, after a hitch of the source
    // age). A Present of any swapchain, a generated one too, ends it, so a
    // generated Present between an evaluation and its frame's Present leaves
    // the wall-clock rule alone.
    // Require mutex. The newest present_ordinal() an evaluation began with
    // (begin_evaluation_locked), the newest one a frame-generation input's
    // evaluation began with or its copy was stamped with, and the Presents
    // DLSS-G adds per real frame that the selection being decided grants
    // (select_provider, selection_policy::generated_presents).
    std::uint64_t newest_evaluation_present{}, frame_generation_present{};
    std::uint32_t presenting_generated{};

    // Present ordinals and frame-generation scope (depth_capture/scope.inl).
    std::uint64_t present_ordinal();
    bool frame_presenting(std::uint64_t ordinal);
    std::uint64_t presented_by(const slot &value);
    fg_retirement *fg_retirement_of(std::uint64_t epoch, std::uint32_t viewport);
    bool frame_generation_role_retired(provider_kind provider, std::uint64_t epoch, std::uint32_t viewport);
    void retire_frame_generation_role(input &value);
    void accept_compute_copy(const slot &value);
    std::uint64_t scope_now();
    evaluation_head *live_frame_generation_scope(std::uint64_t epoch, std::uint32_t viewport,
        std::uint64_t enabled_tick);
    void record_frame_generation_copy(const input &value);
    // Early part declarations (depth_capture/record.inl).
    unsigned provider_index(provider_kind value);
    bool valid_provider(provider_kind value);
    void set_attempt(provider_kind provider, status value);
    bool valid_projection(const sunshine_scene_depth::projection_basis &value) {
      if (value.encoding == sunshine_scene_depth::depth_encoding::linear_distance)
        return std::isfinite(value.raw_scale) && value.raw_scale != 0.0 && std::isfinite(value.raw_bias) &&
          value.direction_supplied && value.reversed == (value.raw_scale < 0.0);
      if (value.encoding != sunshine_scene_depth::depth_encoding::device) return false;
      return !value.supplied || (std::isfinite(value.depth_offset) && std::isfinite(value.depth_scale) &&
        value.depth_scale != 0.0 && std::isfinite(value.raw_scale) && value.raw_scale != 0.0 &&
        std::isfinite(value.raw_bias) && value.reversed == ((value.depth_scale > 0.0) != (value.raw_scale < 0.0)));
    }

    std::uint64_t device_cookie(ID3D12Device *value, bool create = false) {
      return sunshine_native_identity::device_cookie(value, create);
    }

    std::uint64_t source_cookie(ID3D12Object *value) {
      return sunshine_native_identity::resource_cookie(value);
    }
    std::uint64_t retain_source_cookie(ID3D12Object *value) {
      return sunshine_native_identity::resource_cookie(value, true);
    }
    std::uint64_t observed_source_cookie(ID3D12Object *value) {
      // Most transitions concern existing identities. Avoid the creation lock
      // on that hot path, and never overwrite malformed data or driver errors.
      if (!value) return 0;
      std::uint64_t identity{}; UINT size = sizeof(identity);
      const auto result = value->GetPrivateData(sunshine_native_identity::resource_guid, &size, &identity);
      if (SUCCEEDED(result)) return size == sizeof(identity) ? identity : 0;
      return result == DXGI_ERROR_NOT_FOUND ? retain_source_cookie(value) : 0;
    }
    // Depth storage and views: sunshine_depth_format (depth_view_format.h).
    DXGI_FORMAT typeless(DXGI_FORMAT value) { return sunshine_depth_format::storage(value); }
    // The view a depth input is read through: its declared nativeFormat
    // resolves R16_TYPELESS, and only device depth reads an undeclared one.
    DXGI_FORMAT depth_view(DXGI_FORMAT format, const input &value) {
      return sunshine_depth_format::view(format, static_cast<DXGI_FORMAT>(value.resource.native_format),
        value.projection.encoding == sunshine_scene_depth::depth_encoding::device);
    }
    // A single-sample, single-slice 2D texture of a depth format. Its mip
    // chain is irrelevant: only subresource 0 (mip 0) is transitioned and
    // copied, into a single-mip snapshot. Arrays, MSAA and 3D stay refused.
    bool supported_description(const D3D12_RESOURCE_DESC &desc) {
      return desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && desc.SampleDesc.Count == 1 &&
        desc.DepthOrArraySize == 1 && desc.MipLevels >= 1 && desc.Width && desc.Width <= 16384 &&
        desc.Height && desc.Height <= 16384 && typeless(desc.Format) != DXGI_FORMAT_UNKNOWN;
    }
    DXGI_FORMAT diagnostic_storage_format(DXGI_FORMAT format, std::uint32_t declared) {
      // Color/alpha allocations may be typeless while their UI views are typed.
      // Copy those bytes into a typed snapshot of the declared or same-bits
      // UNORM view so both the renderer and diagnostic host receive an
      // explicit format (the UI capture owner checks the same mapping before
      // it records a copy); an unresolved typeless allocation has none.
      return static_cast<DXGI_FORMAT>(auxiliary_snapshot_format(static_cast<std::uint32_t>(format), declared));
    }
    static_assert(auxiliary_snapshot_format(DXGI_FORMAT_R8G8B8A8_TYPELESS) == std::uint32_t(DXGI_FORMAT_R8G8B8A8_UNORM) &&
      auxiliary_snapshot_format(DXGI_FORMAT_B8G8R8A8_TYPELESS) == std::uint32_t(DXGI_FORMAT_B8G8R8A8_UNORM) &&
      auxiliary_snapshot_format(DXGI_FORMAT_R10G10B10A2_TYPELESS) == std::uint32_t(DXGI_FORMAT_R10G10B10A2_UNORM) &&
      auxiliary_snapshot_format(DXGI_FORMAT_R16G16B16A16_TYPELESS, DXGI_FORMAT_R16G16B16A16_FLOAT) ==
        std::uint32_t(DXGI_FORMAT_R16G16B16A16_FLOAT) &&
      !auxiliary_snapshot_format(DXGI_FORMAT_R16G16B16A16_TYPELESS));
    // A8_UNORM (a UIAlpha tag's format) is read by the live renderer only:
    // the D3D11 host cannot open a shared A8_UNORM texture (OpenSharedResource1
    // returns E_INVALIDARG), so a shared (Dump 3D) copy keeps the host's
    // format table (the record path refuses it there).
    unsigned diagnostic_pixel_bytes(DXGI_FORMAT format, std::uint32_t declared) {
      const auto storage = diagnostic_storage_format(format, declared);
      return storage == DXGI_FORMAT_A8_UNORM ? 1u : game3d_debug::pixel_bytes(static_cast<unsigned>(storage));
    }
    // declared: the source's declared view format (Streamline
    // Resource.nativeFormat; 0 none), which resolves a typeless allocation.
    bool diagnostic_description(const D3D12_RESOURCE_DESC &desc, std::uint32_t declared = 0) {
      return desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && desc.SampleDesc.Count == 1 &&
        desc.DepthOrArraySize == 1 && desc.MipLevels == 1 && desc.Width && desc.Width <= 16384 &&
        desc.Height && desc.Height <= 16384 && diagnostic_pixel_bytes(desc.Format, declared) &&
        !(desc.Flags & (D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_VIDEO_DECODE_REFERENCE_ONLY |
          D3D12_RESOURCE_FLAG_VIDEO_ENCODE_REFERENCE_ONLY));
    }
    DXGI_FORMAT auxiliary_copy_format(DXGI_FORMAT format) {
      // Native copies preserve the bytes, including alpha. The renderer uses
      // UNORM views even when the game's corresponding color resource is SRGB.
      switch (format) {
      case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return DXGI_FORMAT_R8G8B8A8_UNORM;
      case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return DXGI_FORMAT_B8G8R8A8_UNORM;
      default: return format;
      }
    }
    // Colour view families: color_family, typeless_color and same_color_family
    // (api_contract.h), shared with the UI capture gate.
    static_assert(color_family(DXGI_FORMAT_R16G16B16A16_UNORM) == DXGI_FORMAT_R16G16B16A16_TYPELESS &&
      color_family(DXGI_FORMAT_A8_UNORM) == DXGI_FORMAT_A8_UNORM && !color_family(DXGI_FORMAT_R8G8B8A8_UINT) &&
      same_color_family(DXGI_FORMAT_R8G8B8A8_TYPELESS, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB) &&
      !same_color_family(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_R8G8B8A8_UNORM) &&
      !same_color_family(DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_D16_UNORM));
    status source_region(const D3D12_RESOURCE_DESC &desc,
        const sunshine_scene_depth::resource_description &resource, copy_region &out) {
      out = {};
      if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || !desc.Width || desc.Width > 16384 ||
          !desc.Height || desc.Height > 16384) return status::malformed;
      const auto area = resource.area;
      // An extent inside the allocation alone defines the region; a declared
      // size that disagrees with the allocation is then ignored (the adapter
      // logs it). A null extent, !(width && height), is the whole allocation,
      // which a declared size must then name (extent_in_allocation).
      if (!sunshine_api_contract::extent_in_allocation(area, desc.Width, desc.Height, resource.width, resource.height))
        return status::malformed;
      // Always retain the full allocation, including packed depth/stencil
      // plane zero. Statistics and the shader apply this same active rectangle;
      // no boxed depth/stencil GPU copy or conversion pass is introduced.
      out.width = static_cast<std::uint32_t>(desc.Width);
      out.height = desc.Height;
      out.area = sunshine_api_contract::normalized_extent(area, out.width, out.height);
      return status::ready;
    }
    status capture_region(const D3D12_RESOURCE_DESC &desc,
        const sunshine_scene_depth::resource_description &resource, copy_region &out) {
      if (!supported_description(desc)) { out = {}; return status::unsupported_resource; }
      return source_region(desc, resource, out);
    }
    // Advanced under mutex (invalidate_all); read without it by the barrier
    // hooks, which keep their recording's payload without the capture lock.
    std::atomic<std::uint64_t> command_generation{1};
  }

  SUNSHINE_CAPTURE_INTERNAL {
    std::array<slot, slot_limit> slots;
    std::array<slot, diagnostic_slot_limit> diagnostic_slots;
    bool diagnostic_slots_active{}; // Protected by mutex; idle callbacks skip this pool.

    template<class Operation> void for_each_capture(Operation operation) {
      for (auto &value : slots) operation(value);
      if (diagnostic_slots_active) for (auto &value : diagnostic_slots) operation(value);
    }
    std::array<std::weak_ptr<const source_reference>, source_limit> sources;
    // Every snapshot storage allocated (allocate_storage), as long as anyone
    // holds it: release_device releases a device's storage whoever holds it
    // (a packet, a ticket). Protected by mutex; expired entries are dropped
    // when it would grow, so it stays as long as the storage alive.
    std::vector<std::weak_ptr<texture_reference>> storage;
    std::array<std::unique_ptr<queue_state>, queue_limit> queues;
    // Protected by mutex: the cookie of the queue whose acquisition is the
    // newest (acquire), the queue that now reads depth; 0 before any. A
    // queue that read depth earlier stops being the reader once another
    // acquires, as when FG moves Present to DLSS-G's queue (read_elsewhere).
    std::uint64_t reader{};
    // Protected by mutex: the place cookie of the newest submission that
    // carried a depth copy of a DIRECT list (submitted), where a list never
    // executed yet is predicted to run (record_impl's copy-out storage); 0
    // before any.
    std::uint64_t last_direct_copy_queue{};
    // FG moves Present to DLSS-G's queue at its first Present, after the
    // render queue's copies of that frame were recorded and submitted while
    // that queue still read depth, so for the source age after FG turns on
    // the reader's own copies are copied out too (U-C1b, reader_may_move).
    // Two windows, each closed once Present moved to another queue since it
    // opened (reader_moves): from FG options turning on
    // (frame_generation_enabled; 0 never), and from the first FG activity
    // of a run (an FG frame's begin, DLSS-G's NGX evaluation:
    // frame_generation_activity) after FG options turned on or after
    // fg_silent_ms without any. The second covers a game that turns FG on
    // away from gameplay (a menu, a loading screen, saved settings at boot),
    // whose first FG Present comes long after the options call, and DLSS-G
    // seen only through NGX, with no options call at all.
    std::atomic<std::uint64_t> frame_generation_on_tick{}, frame_generation_on_moves{};
    std::atomic<std::uint64_t> frame_generation_run_tick{}, frame_generation_run_moves{}, frame_generation_activity_tick{};
    // Changes of the reader (acquire): Present seen on another queue.
    std::atomic<std::uint64_t> reader_moves{};
    // A reader change whose first acquisition had no ready pixels (the flat
    // Present at FG On or Off that U-C1b's copy-outs exist to prevent).
    std::atomic<std::uint64_t> reader_move_unreadable{};
    std::array<evaluation_namespace, provider_count> evaluations;
    // Protected by mutex: the viewports whose FG role FG Off retired, per
    // epoch (retire_source; the oldest gives way).
    std::array<fg_retirement, 4> fg_retirements;

    // Requires mutex. Whether a view's newest evaluation is the frame the
    // Present being decided presents: by its call's stamp, or by its newest
    // copy's (presented_by).
    bool view_presenting(const evaluation_head &view, provider_kind provider) {
      if (frame_presenting(view.present)) return true;
      for (const auto &value : slots)
        if (value.id && !value.preservation_only && value.metadata.provider == provider && value.metadata.epoch == view.epoch &&
            value.metadata.source_id == view.source_id && value.metadata.viewport == view.viewport &&
            value.metadata.sequence == view.sequence && frame_presenting(presented_by(value)))
          return true;
      return false;
    }


    void invalidate(slot &value, capture_failure why) {
      if (value.failure == capture_failure::none) value.failure = why;
      value.invalid = true;
      if (why != capture_failure::producer_signal_failed && why != capture_failure::consumer_signal_failed &&
          why != capture_failure::consumer_capacity && why != capture_failure::consumer_queue_changed)
        value.nomination_invalid = true;
      // An FG copy recorded on a COMPUTE list (DLSS-G's own input, or an FG
      // tag's: S22, N7) whose producer can no longer supply its frame (its
      // recording discarded unsubmitted, as when its queue has no place,
      // replayed, or its queue gone) leaves that frame without an FG copy,
      // exactly as frame_generation_missed records: an accepted copy that
      // never reaches a consumer keeps no SR depth out.
      const auto &m = value.metadata;
      if (value.compute_list && value.source_nominated && m.frame_generation_input &&
          m.provider == provider_kind::streamline && m.epoch && m.sequence &&
          why != capture_failure::consumer_signal_failed && why != capture_failure::consumer_capacity &&
          why != capture_failure::consumer_queue_changed)
        for (auto &view : evaluations[provider_index(m.provider)].views)
          if (view.epoch == m.epoch && view.viewport == m.viewport && view.source_id == m.source_id)
            view.missed_sequence = std::max(view.missed_sequence, m.sequence);
    }

    // Part declarations: functions defined in depth_capture/observation.inl,
    // depth_capture/queues.inl, depth_capture/queue_order_consumer.inl and pool.inl
    // (included at the end of this file) that code here or an earlier part
    // calls or names. A default argument is given here only.
    command_storage::handle command(std::uint64_t native, std::uint64_t cookie, bool create = false);
    void invalidate(command_state &owner, recording_loss why);
    void associate_recording(std::uint64_t native, std::uint64_t *out_cookie);
    source_state *state(command_state &owner, std::uint64_t identity, bool create);
    void invalidate_all();
    void barriers(std::uint64_t native, std::uint64_t cookie, std::uint32_t count, const D3D12_RESOURCE_BARRIER *values);
    void retire_dead_recordings(slot &value);
    void enhanced_textures(std::uint64_t native, std::uint64_t cookie, unsigned count,
        const native_observer::enhanced_texture *textures);
    copy_state::observation observation_of(const source_state *observed);
    state_basis copy_basis(const source_state *observed, const copy_state::decision &chosen);
    bool aliased_after(const command_state &owner, std::uint64_t source, std::uint32_t order);
    void apply_executed(std::uint64_t queue, std::uint32_t count, const native_observer::command_identity *values);
    queue_state *known_queue(std::uint64_t native);
    queue_state *queue(std::uint64_t native);
    bool submitted_success(const slot &value, std::uint64_t native);
    queue_progress producer_progress(const slot &value);
    bool producer_recording_retired(const slot &value);
    bool immutable_for_other_queues(const slot &value);
    const std::shared_ptr<texture_reference> &read_storage(const slot &value);
    bool readable_elsewhere(const slot &value, std::uint64_t consumer, const queue_progress &progress,
        sunshine_queue_order::verdict *relied = nullptr);
    bool reader_live();
    std::uint64_t reader_queue();
    bool uses_queue(const slot &value, std::uint64_t native);
    std::unique_ptr<queue_state> *free_place();
    bool compute_producer_placed();
    bool reader_may_move(std::uint64_t now);
    void read_on(std::uint64_t cookie, bool pixels);
    void collect_retired(std::vector<std::shared_ptr<texture_reference>> &discard);
    bool ordered_after_producer(const slot &value, std::uint64_t consumer, sunshine_queue_order::verdict *relied);
    native_observer::position game_submission_position();
    native_observer::position copy_out_position(const std::array<bool, slot_limit> &copied);
    void set_producer_position(slot &value, bool copied, bool signaled, const native_observer::position &copy_at,
        const native_observer::position &game_at);
    void forget_queue_order(ID3D12Device *device);
    bool rely_on_queue_order(const sunshine_queue_order::verdict &relied, const slot &entry,
        consumer_diagnostic *diagnostic);
    // Recording functions (depth_capture/record.inl).
    evaluation_head nomination_of(const input &value);
    bool same_nomination(const evaluation_head &a, const evaluation_head &b);
    bool matching_evaluation(const evaluation_head &head, const input &value);
    void end_nomination(provider_kind provider, const evaluation_head &nomination);
    void end_nomination(const input &value);
    std::uint64_t record_nomination(const input &value, std::uint64_t normalized_source,
        std::uint64_t cookie, const recording_ref &recording);
    bool reusable_preserved(const slot &entry, const input &value, std::uint64_t recording);
    // Consumer admission, leases and copies (depth_capture/consume.inl).
    void describe_packet(const slot &value, std::uint64_t consumer, packet &out);
    consumer_status preserved_admission(const slot &value, const preservation_ticket &ticket,
        std::uint64_t consumer, std::uint64_t consumer_device, std::uint64_t recording);
    // Capture and provider selection (depth_capture/selection.inl).
    status diagnostic_admission(const slot &value, const queue_progress &progress);
    status submission_status(const slot &value, std::uint64_t native,
        std::uint64_t consumer_device, const queue_progress &progress);
    status nomination_status(const slot &value, std::uint64_t consumer_device);
    status local_texture_admission(const slot &value, std::uint64_t consumer, std::uint64_t device,
        const queue_progress &progress);
    void bind_pixel_consumer(slot &value, std::uint64_t consumer, status pixels);
    capture_pick select_latest_capture(const queue_state &owner, std::uint64_t native,
        std::uint64_t present, provider_kind provider, std::uint64_t now);
    bool pending_frame(const capture_pick &chosen, const queue_state &owner);
    acquisition_decision classify_acquisition(const capture_pick &chosen, const queue_state &owner,
        selection_policy policy);
    depth_choice describe_choice(const capture_pick &chosen, const acquisition_decision &decision,
        const queue_state &owner, selection_policy policy, std::uint64_t present, std::uint64_t tick, bool pixel_ready);
    slot *completed_snapshot(const input &current, std::uint64_t now, std::uint64_t consumer = 0);
    bool same_depth_storage(const input &a, const input &b);
    selection_reason completed_fallback_reason(const queue_state &owner, std::uint64_t present,
        const input &head, const slot &completed, bool compare_storage);
    bool lineage_candidate(const slot &value, const input &fg, std::uint64_t now);
    bool lineage_head(const input &fg);
    slot *lineage_snapshot(const queue_state &owner, std::uint64_t native, std::uint64_t present, const input &head,
        std::uint64_t now);
    const slot *lineage_fallback(const input &fg, std::uint64_t now);
    input head_identity(const evaluation_head &head, provider_kind provider);
    capture_pick select_capture(const queue_state &owner, std::uint64_t native,
        std::uint64_t present, provider_kind provider, std::uint64_t now);
    bool interrupts_admission(const queue_state &owner, const capture_pick &chosen);
    capture_pick select_provider(const queue_state &owner, std::uint64_t native,
        std::uint64_t present, std::uint64_t now, selection_policy policy = {});
    void establish_provider(queue_state &owner, const slot &value, std::uint64_t present);
    void expire_association(queue_state &owner, std::uint64_t now);
    // Pool functions (depth_capture/pool.inl); types and state stay in this shell.
    enum class source_retention;
    struct storage_job;
    struct storage_plan;
    bool current_source_nomination(const slot &value);
    source_retention logical_source_retention(slot &value);
    bool snapshot_storage_retired(const slot &value);
    bool reclaimable(slot &value);
    void release_storage_job(storage_job &job, bool pooled);
    void CALLBACK storage_release_task(PTP_CALLBACK_INSTANCE, void *context);
    void release_storage(std::vector<std::shared_ptr<texture_reference>> &values) noexcept;
    void retire_storage(std::vector<std::shared_ptr<texture_reference>> &discard, std::shared_ptr<texture_reference> &value) noexcept;
    void wait_storage_releases();
    void collect_diagnostics(std::vector<std::shared_ptr<texture_reference>> &discard);
    bool reserved_by_other(const slot &entry, std::uint64_t own);
    void clear_reservation(std::uint64_t token);
    bool live_storage(const slot &entry);
    bool reserve_diagnostic_bytes(std::array<slot, diagnostic_slot_limit> &pool,
        const slot *replacement, std::uint64_t bytes, bool live, std::uint64_t own = 0,
        std::vector<std::shared_ptr<texture_reference>> *discard = nullptr);
    record_stage allocate_storage(const storage_plan &plan, std::shared_ptr<texture_reference> &out);
    bool register_storage(const std::shared_ptr<texture_reference> &value);
    // Protected by mutex: orders the queue places by their last use
    // (queue_state::last_use), so free_place reclaims the longest unused.
    std::uint64_t place_clock{};
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
    bool fail_state_growth{};
#endif
    // The per-resource last-proven state (proven_state.h), under mutex: at
    // most proven_limit textures, the least recently used forgotten first.
    constexpr unsigned proven_limit = 128;
    sunshine_proven_state::table<proven_limit> proven;
    // Depth-stencil textures ReShade destroyed (resource_destroyed), queued
    // without a lock (destroy_resource may run where this owner's lock is
    // held) and forgotten before the table next holds a texture (0: a free
    // slot). A full queue drops one: the table then forgets it as least
    // recently used.
    std::array<std::atomic<std::uint64_t>, proven_limit> destroyed{};
    std::atomic<bool> destroyed_pending{};
    // Requires mutex.
    void forget_destroyed() {
      if (!destroyed_pending.load(std::memory_order_acquire) || !destroyed_pending.exchange(false, std::memory_order_acq_rel)) return;
      for (auto &slot : destroyed)
        if (const auto identity = slot.exchange(0, std::memory_order_acq_rel)) proven.forget(identity);
    }
    // A creation state stops proving a texture once enhanced barriers were
    // seen: ReShade reports a texture created in an initial layout like one
    // created in a legacy state. Set without the lock.
    std::atomic<bool> enhanced_barriers_seen{};
    struct {
      std::atomic<std::uint64_t> unrecorded{}, unreported{}, invalidated{}, missed_reset{}, uncovered{}, changed_copies{};
      // Copies of a latched source with nothing on their recording (D6, N8).
      std::atomic<std::uint64_t> earlier_copied{}, earlier_refused{};
    } proven_counters;
    // Copy-outs (copy_out): made, and copies left to the Reset rule because
    // every list of the queue's ring was in flight or a list call failed, or
    // (failed too) their copy-out storage could not be allocated.
    // U-C1b: copies recorded with storage because their list was not known
    // to execute on the reader's queue (predicted), copies given an idle
    // slot's cached storage at their submission (taken), copies submitted
    // on another queue than the reader's with no storage at all, left to
    // the Reset rule (unstored: recorded for the reader's queue before
    // Present moved, with no idle storage of their shape), and predicted
    // storage the idle sweep released unused (released), and copies that sought no storage
    // of their own because an allocation of their shape had just failed
    // (skipped, copy_out_failure).
    struct {
      std::atomic<std::uint64_t> made{}, ring_busy{}, failed{}, predicted{}, taken{}, unstored{}, released{}, skipped{};
    } copy_out_counters;
    // Protected by mutex: the last copy-out storage allocation that failed
    // (record_impl), by device, shape and format. For the source age after
    // it, a copy of that shape does not allocate copy-out storage again on
    // the game's recording thread (under budget pressure the driver may try
    // to make room before it fails, at every copy of every frame with FG
    // On); copy_out may still give it an idle slot's storage at submission.
    struct {
      std::uint64_t device_identity{}, width{}, height{}, tick{};
      DXGI_FORMAT format{};
    } copy_out_failure;
    // Reads admitted on another queue by the queue-order model before their
    // producer fence completed (U-C1; the queue order line's proofs). Only
    // depth_capture/queue_order_consumer.inl writes and reads it.
    std::atomic<std::uint64_t> order_inflight_reads{};
    // The shared source age bounds nomination, completed-snapshot admission and
    // liveness alike (freshness.h).
    using sunshine_freshness::within_source_age;
    // states_covered's answers per method table (depth_capture/observation.inl).
    struct covered_table { std::uintptr_t table{}; std::uint64_t generation{}; bool covered{}; };
    std::array<covered_table, 8> covered_tables{};
    unsigned covered_next{}, coverage_checks{};
    std::uint64_t coverage_window{};
    // The capture owner's safety bounds that refused work (capacity_report),
    // one bit each, set without the lock.
    enum capacity_bound : unsigned { queue_places = 1, source_table = 2, snapshot_pool = 4, auxiliary_pool = 8 };
    std::atomic<unsigned> capacity_hits{}, capacity_reported{};
    // Part declaration (depth_capture/queues.inl), after its type.
    void capacity_refused(capacity_bound bound);
    // Protected by mutex: an idle queue's place was reclaimed (free_place), so
    // a queue with no place may carry a copy (register_compute_producer).
    bool places_reclaimed{};
    // register_compute_producer's place rule for these: depth_capture/queues.inl.
    bool compute_copies_recorded{}; // Protected by mutex: a copy was ever recorded on a COMPUTE list.
    bool compute_unplaced{}; // Protected by mutex.
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
    // SunshineStreamlineTestRefuseCopyOuts: while set, copy_out refuses as if
    // every list of the queue's ring were in flight, so the copies keep the
    // Reset rule (the NGX cross-queue regression's no-copy-out variant).
    std::atomic<bool> refuse_copy_outs{};
#endif
    enum class source_retention { none, current_nomination, completed_fallback };
    using storage_list = std::vector<std::shared_ptr<texture_reference>>;
    // Capture storage the present thread (poll, a Present's observation) or a
    // game thread (a record call, a queue event) drops is released on a pool
    // thread: the last Release of a large committed resource frees its video
    // memory in the driver and can wait on it (bc9989d0: 17 and 81 ms for the
    // display's storage; the fg4k contract matrix of 10-27: an 8 ms "depth
    // input observation" at 4K with the idle sweep's releases inline). The
    // jobs are counted: release_device and shutdown wait until none is
    // outstanding, so none outlives its device or the add-on.
    struct storage_jobs_state {
      std::mutex mutex;
      std::condition_variable idle;
      unsigned outstanding{};
    } storage_jobs;
    struct storage_job {
      storage_list values;
      DWORD caller{};
    };
    // Storage dropped under the capture lock goes to the pool once the lock
    // is released: declared before the lock, destroyed after it.
    struct pooled_storage : storage_list {
      pooled_storage() = default;
      pooled_storage(const pooled_storage &) = delete;
      pooled_storage &operator=(const pooled_storage &) = delete;
      ~pooled_storage() { release_storage(*this); }
    };
    constexpr std::uint64_t reservation_timeout_ms = 1000;
    // Reservation tokens never consume capture IDs (serial). Protected by mutex.
    std::uint64_t reservation_serial{};
    // Live snapshots (record_local_texture, reused every frame) and Dump 3D
    // copies keep separate budgets, so an armed dump never evicts or crowds
    // live storage. Dump copies keep the fixed limit. Live storage is sized
    // from the request: room for this many snapshots of its size (a few UI
    // kinds, each with a frame in flight), never less than the fixed limit.
    constexpr std::uint64_t live_snapshot_reserve = 8;
    // Storage a record attempt allocates without the capture lock.
    struct storage_plan {
      ID3D12Device *device{};
      std::uint64_t device_identity{}, allocation_bytes{};
      D3D12_RESOURCE_DESC desc{};
      D3D12_RESOURCE_STATES state{D3D12_RESOURCE_STATE_COMMON};
      // view: the depth SRV format created now (UNKNOWN: a heap only, for a
      // lease's SRV created later).
      DXGI_FORMAT view{DXGI_FORMAT_UNKNOWN};
      bool shared{}, descriptors{};
    };
  }

  namespace {
    void observed_submission(std::uint64_t queue, std::uint32_t count, const native_observer::command_identity *values);
  }
  void initialize(bool enabled) {
    requested = false;
    {
      std::lock_guard lock(mutex);
      evaluations = {};
      for (auto &attempt : attempt_status) attempt = status::unavailable;
      for (auto &owner : queues) if (owner) {
        owner->provider_established = false;
        owner->source_id = owner->last_epoch = owner->last_sequence = owner->last_source_tick = owner->nominated_epoch = owner->present_capture = 0;
        owner->admission_after = 0;
      }
    }
    requested = enabled;
    reason = enabled ? status::unavailable : status::inactive;
    native_observer::callbacks callbacks;
    callbacks.barriers = barriers;
    callbacks.submitted = observed_submission; callbacks.invalidated = invalidate_all;
    callbacks.enhanced_textures = enhanced_textures;
    callbacks.associate_recording = associate_recording;
    native_observer::initialize(callbacks);
    native_observer::set_active(enabled);
  }
  bool active() {
    if (!requested.load()) return false;
    std::lock_guard lock(mutex);
    return std::any_of(queues.begin(), queues.end(), [](const auto &value) { return value && !value->retiring; });
  }
  namespace {
    // Capture admissions, for the five-second report.
    struct {
      std::atomic<std::uint64_t> covered{}, unknown{}, closed{}, pass{}, states_observed{}, states_declared{};
    } list_counts;
    enum class list_view { unknown, open, closed, pass };
    // Requires mutex. Coverage: ReShade's lifecycle (or a registered runtime
    // list's submissions) reports the recording open and outside a render
    // pass. Bundles cannot record barriers, clears or copies, so executing
    // one changes no resource state and leaves coverage alone.
    list_view lifecycle_view(const command_state *owner) {
      if (!owner || !owner->reported) return list_view::unknown;
      return owner->closed ? list_view::closed : owner->render_pass ? list_view::pass : list_view::open;
    }
    void count_admission(list_view view, bool states_observed) {
      constexpr auto relaxed = std::memory_order_relaxed;
      auto &c = list_counts;
      if (view == list_view::open) (states_observed ? c.states_observed : c.states_declared).fetch_add(1, relaxed);
      (view == list_view::open ? c.covered : view == list_view::unknown ? c.unknown :
        view == list_view::closed ? c.closed : c.pass).fetch_add(1, relaxed);
    }
    // ReShade's own immediate lists, which its list events never report.
    SRWLOCK runtime_lock = SRWLOCK_INIT;
    std::array<std::uint64_t, 8> runtime_lists{}; // Guarded by runtime_lock.
    std::atomic<bool> runtime_lists_known{};
    // Part declarations (depth_capture/observation.inl), in this namespace.
    bool runtime_list(std::uint64_t native);
    void lifecycle_recording(std::uint64_t native, list_event event);
    std::atomic<std::uint64_t> game_execution_count{};
    // The newest game execution (its game_execution_count) per queue, as the
    // submission hook reports it. The queue places' safety bound, not a
    // budget: a queue with no entry takes a free one, else the entry of the
    // queue whose newest execution is oldest, and a retired queue's entry is
    // freed (retire_queue). So no number of queues, nor their churn over a
    // session, leaves an executing queue without its record; a queue whose
    // entry was taken records again at its next execution, and until then a
    // deferred copy that needs it waits. Its 8 entries were once claimed for
    // good, so a presenting queue beyond them, or one recreated after them,
    // never recorded; Assassin's Creed Shadows (10-25) created 9 queues, its
    // presenting queue the 7th and one recreated mid-session.
    // The entries in use are the first queue_execution_count, so a lookup
    // scans only the queues that execute. A known queue's submission raises
    // its newest execution under a shared hold (atomically, as submissions
    // race), so only a queue's first one, or one after its entry was taken,
    // holds the lock exclusively. Readers read newest under a shared hold.
    struct queue_execution { std::uint64_t queue{}, newest{}; };
    std::array<queue_execution, queue_limit> queue_executions{};
    std::size_t queue_execution_count{};
    SRWLOCK queue_execution_lock = SRWLOCK_INIT;
    // Part declarations (depth_capture/queues.inl), in this namespace.
    queue_execution *find_queue_execution(std::uint64_t queue);
    void forget_queue_execution(std::uint64_t queue);
  }
  list_coverage_counts list_coverage() {
    list_coverage_counts out;
    const auto &c = list_counts;
    out.covered = c.covered.load(); out.unknown = c.unknown.load(); out.closed = c.closed.load();
    out.pass = c.pass.load();
    out.states_observed = c.states_observed.load(); out.states_declared = c.states_declared.load();
    return out;
  }
  bool capacity_report(char *out, std::size_t size) {
    const auto hits = capacity_hits.load(std::memory_order_relaxed);
    if (!(hits & ~capacity_reported.load(std::memory_order_relaxed))) return false;
    const auto fresh = hits & ~capacity_reported.fetch_or(hits, std::memory_order_relaxed);
    if (!fresh || !out || !size) return false;
    const auto named = [&](unsigned bit) { return (fresh & bit) != 0; };
    const auto bound = [&](unsigned bit, const char *text, unsigned limit) {
      if (!named(bit)) return;
      const auto used = std::strlen(out);
      if (used < size) std::snprintf(out + used, size - used, " %s=%u", text, limit);
    };
    std::snprintf(out, size, "Sunshine capture capacity: refused at a safety bound:");
    bound(queue_places, "queue_places", queue_limit);
    bound(source_table, "source_table", source_limit);
    bound(snapshot_pool, "depth_snapshots", pixel_slot_limit);
    bound(auxiliary_pool, "auxiliary_snapshots", diagnostic_slot_limit);
    const auto used = std::strlen(out);
    if (used < size)
      std::snprintf(out + used, size - used, "; every place or entry was in use (queue places: each queue names a "
        "capture, a provider or an unfinished fence), so that capture was refused as capacity; logged once per bound");
    return true;
  }
  bool list_coverage_report(std::uint64_t now_ms, char *out, std::size_t size) {
    static std::atomic<std::uint64_t> next_log{}, logged_admissions{};
    auto next = next_log.load(std::memory_order_relaxed);
    if (now_ms < next || !next_log.compare_exchange_strong(next, now_ms + 5000, std::memory_order_relaxed)) return false;
    const auto c = list_coverage();
    const auto admissions = c.covered + c.unknown + c.closed + c.pass;
    if (admissions == logged_admissions.exchange(admissions, std::memory_order_relaxed)) return false;
    const auto u = [](std::uint64_t value) { return static_cast<unsigned long long>(value); };
    std::snprintf(out, size,
      "Sunshine list lifecycle: admissions covered=%llu (states observed=%llu declared=%llu) not_open={unknown=%llu closed=%llu pass=%llu}",
      u(c.covered), u(c.states_observed), u(c.states_declared), u(c.unknown), u(c.closed), u(c.pass));
    return true;
  }
  namespace {
    // A released device's objects whose GPU work had not finished
    // (release_device), kept for good: under ReShade each resource of a
    // destroyed device calls into that device as it is destroyed, so none may
    // be released after it. Protected by mutex; never freed (the add-on is
    // pinned).
    struct kept_objects {
      std::vector<std::shared_ptr<texture_reference>> storage;
      std::vector<source_ref> sources;
      std::vector<std::unique_ptr<queue_state>> places;
    };
    kept_objects *kept{};
    // release_device's bound on the wait for a device's queues, which the game
    // has drained before it releases its device. A safety bound, not a
    // tuning: it keeps a queue that never reaches its fence (one waiting on a
    // fence nobody signals) from hanging the game's release, and GPU work
    // still unfinished after it is never released.
    constexpr std::uint64_t release_wait_bound_ms = 10000;
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
    std::atomic<std::uint64_t> release_wait_ms{release_wait_bound_ms};
#else
    constexpr std::uint64_t release_wait_ms = release_wait_bound_ms;
#endif
  }
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
  namespace testing {
  }
#endif
  namespace {
    // Idle snapshot storage kept for reuse beyond this is released.
    constexpr std::uint64_t idle_pixel_byte_limit = 256ull * 1024 * 1024;
    // Cached colour/mask storage a live snapshot has not reused for this long.
    constexpr std::uint64_t idle_live_storage_ms = 2000;
  }

  SUNSHINE_CAPTURE_INTERNAL {
    // frame_generation_scope_live's clock. The seam characterization fixes it
    // to pin the scope's exact edges (streamline_depth_capture_test_access.h).
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
    std::atomic<std::uint64_t> scope_clock{};
#else
#endif
  }
  SUNSHINE_CAPTURE_INTERNAL {
  }



  std::uint64_t game_executions_on(std::uint64_t queue) {
    com_ptr<ID3D12CommandQueue> checked;
    if (!queue || !query_native(queue, IID_ID3D12CommandQueue, checked)) return 0;
    queue = reinterpret_cast<std::uint64_t>(checked.p); // As the submission hook reports it.
    std::uint64_t newest{};
    AcquireSRWLockShared(&queue_execution_lock);
    if (auto *entry = find_queue_execution(queue))
      newest = std::atomic_ref<std::uint64_t>(entry->newest).load(std::memory_order_relaxed);
    ReleaseSRWLockShared(&queue_execution_lock);
    return newest;
  }



  namespace {
  }

#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
  namespace {
    // One Present of an earlier frame landing inside the next nominated copy's
    // call, between its input's read and the copy (record_impl).
    std::atomic<bool> present_before_next_copy{};
  }
  extern "C" __declspec(dllexport) void SunshineStreamlineTestPresentBeforeNextCopy() {
    present_before_next_copy.store(true, std::memory_order_release);
  }

  // Withdraws that Present when no nominated copy took it (the session
  // timelines' Present inside an SR evaluation that nominated none); whether
  // it was still pending.
  extern "C" __declspec(dllexport) BOOL SunshineStreamlineTestCancelPresentBeforeNextCopy() {
    return present_before_next_copy.exchange(false, std::memory_order_acq_rel) ? TRUE : FALSE;
  }

  // Whether every consumer of a live capture reads its copy-out storage
  // (read_storage), not the snapshot its game recording may still replay
  // into: the NGX cross-queue regression's proof of what it admitted.
  extern "C" __declspec(dllexport) BOOL SunshineStreamlineTestCaptureReadsCopyOut(std::uint64_t capture, BOOL *copy_out) {
    if (!capture || !copy_out) return FALSE;
    std::lock_guard lock(mutex);
    for (const auto &value : slots)
      if (value.id == capture) { *copy_out = value.copied_out && value.published ? TRUE : FALSE; return TRUE; }
    return FALSE;
  }
  // While refuse is set, no copy-out is made (counted as ring_busy, every
  // list of the queue's ring in flight): another queue reads a snapshot only
  // once its recording retired (the Reset rule, immutable_for_other_queues).
  extern "C" __declspec(dllexport) void SunshineStreamlineTestRefuseCopyOuts(BOOL refuse) {
    refuse_copy_outs.store(refuse != FALSE, std::memory_order_release);
  }
  namespace {
    std::atomic<HANDLE> nomination_gate_entered{}, nomination_gate_release{};
    void wait_nomination_gate() {
      if (const auto entered = nomination_gate_entered.exchange(nullptr, std::memory_order_acq_rel)) {
        const auto release = nomination_gate_release.exchange(nullptr, std::memory_order_acq_rel);
        SetEvent(entered);
        if (release) WaitForSingleObject(release, 5000);
      }
    }
  }
  extern "C" __declspec(dllexport) void SunshineStreamlineTestNominationGate(HANDLE entered, HANDLE release) {
    nomination_gate_release.store(release, std::memory_order_relaxed);
    nomination_gate_entered.store(entered, std::memory_order_release);
  }
  namespace {
    // The newest nominated pixel copy published (record_impl): its capture
    // id, with the frame-generation-input flag in the top bit.
    std::atomic<std::uint64_t> newest_published_copy{};
  }
  // The capture id of the newest nominated pixel copy the add-on published,
  // and whether it is an FG scope's input (false: none yet). The session
  // timelines read it after each call that may copy, so their freshness
  // rule names each copy as frame_record::depth_capture does
  // (test_game3d_continuity.h).
  extern "C" __declspec(dllexport) BOOL SunshineStreamlineTestNewestCopy(std::uint64_t *id, BOOL *frame_generation_input) {
    const auto value = newest_published_copy.load(std::memory_order_acquire);
    if (id) *id = value & ~(1ull << 63);
    if (frame_generation_input) *frame_generation_input = (value >> 63) ? TRUE : FALSE;
    return value ? TRUE : FALSE;
  }
#endif

#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
  namespace testing {
    std::atomic<void (*)(void *)> allocation_hook{};
    std::atomic<void *> allocation_context{};
    void set_allocation_hook(void (*hook)(void *), void *context) {
      allocation_context.store(context);
      allocation_hook.store(hook);
    }
    std::atomic<unsigned> copy_out_allocation_failures{};
    void fail_copy_out_allocations(unsigned count) { copy_out_allocation_failures.store(count); }
    void settle_storage_releases() { wait_storage_releases(); }
    bool fail_copy_out_allocation() {
      for (auto count = copy_out_allocation_failures.load(); count;)
        if (copy_out_allocation_failures.compare_exchange_weak(count, count - 1)) return true;
      return false;
    }
  }
#endif

#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
#endif
  proven_state_counts proven_counts() {
    proven_state_counts out;
    {
      std::lock_guard lock(mutex);
      out.tracked = proven.size();
      out.latched = proven.latched_count();
      out.latch_lost = proven.latch_lost();
    }
    const auto &c = proven_counters;
    constexpr auto relaxed = std::memory_order_relaxed;
    out.unrecorded = c.unrecorded.load(relaxed); out.unreported = c.unreported.load(relaxed);
    out.invalidated = c.invalidated.load(relaxed); out.missed_reset = c.missed_reset.load(relaxed);
    out.uncovered = c.uncovered.load(relaxed); out.changed_copies = c.changed_copies.load(relaxed);
    out.earlier_copied = c.earlier_copied.load(relaxed); out.earlier_refused = c.earlier_refused.load(relaxed);
    out.copied_out = copy_out_counters.made.load(relaxed); out.copy_out_busy = copy_out_counters.ring_busy.load(relaxed);
    out.copy_out_failed = copy_out_counters.failed.load(relaxed);
    out.copy_out_predicted = copy_out_counters.predicted.load(relaxed); out.copy_out_taken = copy_out_counters.taken.load(relaxed);
    out.copy_out_released = copy_out_counters.released.load(relaxed);
    out.copy_out_unstored = copy_out_counters.unstored.load(relaxed);
    out.copy_out_skipped = copy_out_counters.skipped.load(relaxed);
    out.reader_moves = reader_moves.load(relaxed);
    out.reader_move_unreadable = reader_move_unreadable.load(relaxed);
    return out;
  }

#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
#endif
  SUNSHINE_CAPTURE_INTERNAL {
  }
  status last_status() { return reason.load(); }
  const char *name(status value) {
    switch (value) {
#define CASE(x) case status::x: return #x
    CASE(inactive); CASE(unavailable); CASE(malformed); CASE(unsupported_state); CASE(missing_state);
    CASE(conflicting_state); CASE(incomplete_state); CASE(unsupported_lifetime);
    CASE(unsupported_resource); CASE(unsupported_queue); CASE(exhausted); CASE(recorded); CASE(submitted);
    CASE(ready); CASE(stale); CASE(ambiguous); CASE(failed);
#undef CASE
    }
    return "unknown";
  }
  const char *name(capture_failure value) {
    switch (value) {
#define CASE(x) case capture_failure::x: return #x
    CASE(none); CASE(observer_loss); CASE(discarded_recording); CASE(close_failed); CASE(replay);
    CASE(producer_signal_failed); CASE(producer_queue_changed); CASE(consumer_signal_failed); CASE(consumer_queue_changed);
    CASE(evaluation_failed); CASE(queue_retired); CASE(consumer_capacity); CASE(source_retired);
    CASE(proven_state_changed);
#undef CASE
    }
    return "unknown";
  }
  const char *name(selection_reason value) {
    switch (value) {
#define CASE(x) case selection_reason::x: return #x
    CASE(not_attempted); CASE(no_current_nomination); CASE(inactive_views); CASE(ambiguous_views);
    CASE(pending_nomination); CASE(current_capture); CASE(current_already_consumed); CASE(no_admissible_capture);
    CASE(no_completed_snapshot); CASE(completed_before_gap); CASE(completed_not_older);
    CASE(completed_already_consumed); CASE(presentation_already_selected); CASE(format_changed);
    CASE(completed_not_readable); CASE(completed_snapshot); CASE(fg_scope_missing); CASE(fg_scope_mismatch);
    CASE(lineage_snapshot); CASE(completed_older_than_shown);
#undef CASE
    }
    return "unknown";
  }
  const char *name(state_basis value) {
    switch (value) {
#define CASE(x) case state_basis::x: return #x
    CASE(unknown); CASE(observed_legacy); CASE(observed_enhanced); CASE(declared); CASE(contract); CASE(proven);
#undef CASE
    }
    return "unknown";
  }
  const char *name(consumer_status value) {
    switch (value) {
#define CASE(x) case consumer_status::x: return #x
    CASE(not_attempted); CASE(ready); CASE(missing_input); CASE(unsupported_interface); CASE(observer_not_ready);
    CASE(missing_cookie); CASE(unsupported_command_type); CASE(missing_device); CASE(missing_device_identity);
    CASE(device_mismatch); CASE(recording_missing); CASE(recording_closed); CASE(recording_invalid);
    CASE(recording_render_pass); CASE(slot_missing); CASE(ownership_mismatch); CASE(consumer_capacity);
    CASE(capture_not_ready); CASE(invalid_destination);
#undef CASE
    }
    return "unknown";
  }
  const char *name(recording_loss value) {
    switch (value) {
#define CASE(x) case recording_loss::x: return #x
    CASE(none); CASE(global_observation_loss); CASE(source_identity_unavailable);
    CASE(source_state_capacity); CASE(close_failed);
#undef CASE
    }
    return "unknown";
  }
  const char *name(record_stage value) {
    switch (value) {
#define CASE(x) case record_stage::x: return #x
    CASE(not_attempted); CASE(inactive); CASE(malformed_input); CASE(missing_source);
    CASE(unsupported_lifetime); CASE(unsupported_proof); CASE(command_interface); CASE(source_identity);
    CASE(observer_coverage); CASE(command_type); CASE(command_device); CASE(device_identity); CASE(source_lifetime);
    CASE(recording_missing); CASE(recording_closed); CASE(recording_invalid); CASE(recording_render_pass);
    CASE(incomplete_state); CASE(missing_state); CASE(conflicting_state); CASE(unsupported_state); CASE(resource_region);
    CASE(capacity); CASE(texture_allocation); CASE(descriptor_allocation); CASE(recorded);
    CASE(source_interface); CASE(source_device); CASE(source_description); CASE(source_cookie); CASE(retained);
#undef CASE
    }
    return "unknown";
  }
}

#include "depth_capture/consume.inl"
#include "depth_capture/scope.inl"
#include "depth_capture/selection.inl"
#include "depth_capture/record.inl"
#include "depth_capture/observation.inl"
#include "depth_capture/queues.inl"
#include "depth_capture/queue_order_consumer.inl"
#include "depth_capture/pool.inl"
