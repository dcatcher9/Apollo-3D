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
    std::uint64_t present_ordinal() { return presents.load(std::memory_order_acquire) + 1; }
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
    // Requires mutex. Whether a stamp is the frame the Present being decided
    // presents: no other Present came since its call (above), or, for a
    // frame that made a frame-generation call (its DLSS-G call is what adds
    // Presents for it), one of the `generated` Presents after its first, as
    // long as no later evaluation began. With FG on DLSS-G presents a frame's generated
    // Presents and its real one in either order, so a hitch between a
    // frame's evaluation and its last Present made every Present after its
    // first mono (inactive_views, retained_depth_expired; the 10-27 review's
    // F1 and F2, open items 1 and 2), and the first Present after the pause
    // without depth restarted the stereo fade. At most generated + 1
    // Presents count, ended early by the next frame's first evaluation, so a
    // game that keeps presenting without evaluating reaches the source age
    // at its (generated + 2)-th Present as before.
    bool frame_presenting(std::uint64_t ordinal) {
      if (!ordinal) return false;
      const auto n = presents.load(std::memory_order_acquire);
      if (ordinal >= n) return true;
      const std::uint64_t generated = ordinal == frame_generation_present ? presenting_generated : 0;
      return generated && ordinal >= newest_evaluation_present && n - ordinal <= generated;
    }

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

    // The present ordinal of the frame a copy is: its submission's on a
    // presenting queue when seen (slot::submitted_present), else its call's.
    std::uint64_t presented_by(const slot &value) {
      return value.submitted_present ? value.submitted_present : value.present;
    }
    // Requires mutex. The viewport's FG retirement in epoch, or null.
    fg_retirement *fg_retirement_of(std::uint64_t epoch, std::uint32_t viewport) {
      if (!epoch) return nullptr;
      for (auto &mark : fg_retirements) if (mark.epoch == epoch && mark.viewport == viewport) return &mark;
      return nullptr;
    }
    // Requires mutex. While FG Off keeps a viewport's FG role retired (no FG
    // On since), no input is the viewport's FG input: a copy of a frame from
    // before the Off that reaches the capture owner after it (a deferred FG
    // tag copy made at a late Off's Present, DLSS-G's evaluation racing the
    // options call) is the frame's own depth of the display lineage, so it
    // is recorded and evaluated as the viewport's ordinary source (source
    // 0), and every evaluation of the viewport supersedes its FG view. Until
    // 10-28 such a copy became the viewport's FG view again beside its SR
    // view: every Present read two active views (ambiguous_views), moved the
    // gap watermark over each readable SR copy and held the stale FG copy
    // until it expired (the 10-28 review's P1).
    bool frame_generation_role_retired(provider_kind provider, std::uint64_t epoch, std::uint32_t viewport) {
      if (provider != provider_kind::streamline) return false;
      const auto *mark = fg_retirement_of(epoch, viewport);
      return mark && mark->active;
    }
    // Requires mutex. value's metadata with its FG role retired: the
    // viewport's ordinary source.
    void retire_frame_generation_role(input &value) {
      if (value.frame_generation_input && frame_generation_role_retired(value.provider, value.epoch, value.viewport)) {
        value.frame_generation_input = false;
        value.source_id = 0;
      }
    }
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
    status diagnostic_admission(const slot &value, const queue_progress &progress) {
      if (value.diagnostic_released) return status::stale;
      if (value.invalid || (value.finished && !value.success)) return status::failed;
      if (!value.finished || !value.producer_submitted) return status::recorded;
      if (!progress.valid() || !value.producer_fence || value.retirement_unknown) return status::failed;
      if (!producer_recording_retired(value) || progress.completed < value.producer_fence) return status::submitted;
      return status::ready;
    }
    status submission_status(const slot &value, std::uint64_t native,
        std::uint64_t consumer_device, const queue_progress &progress) {
      if (!submitted_success(value, value.queue))
        return value.invalid || (value.finished && !value.success) ? status::failed : status::recorded;
      if (value.nomination_only) return value.pixel_failure;
      // Consumers temporarily transition the owned copy for their own copies.
      // Bind a slot to one consuming timeline instead of permitting unordered
      // state transitions on multiple queues. Actual retirement stays separate.
      if (value.consumer_queue && value.consumer_queue != native) return status::unsupported_queue;
      if (value.queue == native) return status::ready; // Existing queue ordering, no completion query required.
      if (!consumer_device || progress.device_identity != consumer_device) return status::unsupported_queue;
      if (!progress.valid()) return status::failed;
      // A completed but still-closed game recording may legally replay its
      // injected copy, and post-Execute replay detection would be too late.
      // Its copy-out cannot be written again; without one, only the
      // recording's Reset/destruction makes the copy immutable for a foreign
      // queue (immutable_for_other_queues).
      if (!immutable_for_other_queues(value)) return status::submitted;
      // The game may already order this producer after future consumer work.
      // Adding a reverse GPU Wait here can form a cycle, even after flushing
      // the consumer's immediate list. Poll completion, or read the game's own
      // queue order (U-C1), without adding an edge to the game's queue graph;
      // pending source metadata remains admissible.
      return readable_elsewhere(value, native, progress) ? status::ready : status::submitted;
    }
    status nomination_status(const slot &value, std::uint64_t consumer_device) {
      if (!value.source_nominated || !value.id || !value.finished || !value.success || !value.producer_submitted)
        return value.invalid || (value.finished && !value.success) ? status::failed : status::recorded;
      // A failed snapshot fence/consumer concerns pixels, not the already
      // observed successful evaluation and real producer submission.
      if (value.nomination_invalid) return status::failed;
      return value.metadata.source && consumer_device && value.metadata.source->device_identity == consumer_device ?
        status::ready : status::unsupported_queue;
    }
    status local_texture_admission(const slot &value, std::uint64_t consumer, std::uint64_t device,
        const queue_progress &progress) {
      if (!value.diagnostic_only || !value.diagnostic_reusable) return status::unsupported_resource;
      if (value.diagnostic_released) return status::stale;
      if (!value.texture || value.texture->device_identity != device) return status::unsupported_queue;
      if (value.retirement_unknown) return status::failed;
      // Reuse the depth owner's existing same-queue ordering contract. A later
      // producer replay on this queue follows an already submitted consumer;
      // it still invalidates future acquisition and retains every GPU lease.
      return submission_status(value, consumer, device, progress);
    }
    void bind_pixel_consumer(slot &value, std::uint64_t consumer, status pixels) {
      // Source authority may be admitted even when this queue cannot read the
      // pixels. Such a nomination must never steal an existing read timeline.
      if (pixels == status::ready && !value.nomination_only &&
          (!value.consumer_queue || value.consumer_queue == consumer)) value.consumer_queue = consumer;
    }
    // The shared source age bounds nomination, completed-snapshot admission and
    // liveness alike (freshness.h).
    using sunshine_freshness::within_source_age;
    capture_pick select_latest_capture(const queue_state &owner, std::uint64_t native,
        std::uint64_t present, provider_kind provider, std::uint64_t now) {
      capture_pick out;
      out.provider = provider;
      const auto &current = evaluations[provider_index(provider)];
      // The newest nomination's view, active or not (capture_pick::named).
      for (const auto &value : current.views)
        if (current.epoch && current.sequence && value.epoch == current.epoch && value.sequence == current.sequence &&
            value.source_id == current.source_id) out.named = &value;
      const auto is_latest = [&](const slot &value) {
        return value.id && !value.preservation_only && value.metadata.provider == provider && value.metadata.epoch == current.epoch &&
          value.metadata.sequence == current.sequence && value.metadata.source_id == current.source_id;
      };
      for (const auto &value : slots) if (is_latest(value) && (!out.latest || value.id > out.latest->id)) out.latest = &value;
      out.selection = out.latest ? selection_reason::no_admissible_capture : selection_reason::no_current_nomination;
      if (out.latest && out.latest->queue != native) out.progress = producer_progress(*out.latest);
      // Active: its newest evaluation is younger than the source age, or is
      // the frame this Present presents (frame_presenting), by its call or by
      // its newest copy's submission (presented_by).
      unsigned active_views = 0;
      for (const auto &value : current.views)
        if (value.epoch == current.epoch && value.sequence &&
            (sunshine_freshness::fresh_lt(value.tick, now, sunshine_freshness::source_age_ms) || view_presenting(value, provider)))
          ++active_views;
      if (active_views != 1) {
        out.result = active_views ? status::ambiguous : status::stale;
        out.selection = active_views ? selection_reason::ambiguous_views : selection_reason::inactive_views;
        return out;
      }
      // The newest nomination's view, when it is that one active view: its
      // viewport names the head's display lineage (select_capture).
      for (const auto &value : current.views)
        if (current.epoch && current.sequence && value.epoch == current.epoch && value.sequence == current.sequence &&
            value.source_id == current.source_id &&
            (sunshine_freshness::fresh_lt(value.tick, now, sunshine_freshness::source_age_ms) || view_presenting(value, provider)))
          out.head = &value;
      const auto &pending = current.pending_nomination;
      if (pending.sequence && pending.epoch == current.epoch && pending.sequence == current.sequence &&
          pending.source_id == current.source_id &&
          (sunshine_freshness::fresh_lt(pending.tick, now, sunshine_freshness::source_age_ms) || frame_presenting(pending.present)) &&
          owner.provider_established && owner.provider == provider && owner.source_id == pending.source_id &&
          owner.viewport == pending.viewport && owner.last_epoch == pending.epoch && pending.sequence > owner.last_sequence)
        out.pending_nomination = &pending;
      // Intermediate metadata-only tickets do not finish a multi-tag attempt.
      // Its remaining candidate may still produce current pixels. Keep the
      // bounded pending identity until the single batch scope has resolved.
      if (out.pending_nomination) {
        // No candidate is final yet. Do not let a rejected intermediate slot
        // overwrite the clean pending identity with its failure diagnostics.
        out.latest = nullptr;
        out.result = status::recorded;
        out.selection = selection_reason::pending_nomination;
        return out;
      }
      for (auto &value : slots) {
        if (!is_latest(value) || value.id <= owner.admission_after) continue;
        const auto progress = value.queue == native ? queue_progress{} : producer_progress(value);
        if ((value.source_nominated ? nomination_status(value, owner.device_identity) :
            submission_status(value, native, owner.device_identity, progress)) != status::ready) continue;
        if (owner.present == present && owner.present_capture) {
          if (value.id == owner.present_capture) { out.value = &value; out.progress = progress; break; }
          continue;
        }
        // The consumed frame, also of an FG role FG Off retired (retire_source).
        if (owner.provider == provider && owner.last_epoch == value.metadata.epoch &&
            value.metadata.sequence <= owner.last_sequence) continue;
        if (!out.value || value.id > out.value->id) { out.value = &value; out.progress = progress; }
      }
      if (!out.value) {
        out.result = out.pending_nomination ? status::recorded : attempt_status[provider_index(provider)].load();
        if (out.latest) {
          if (out.latest->invalid || (out.latest->finished && !out.latest->success)) out.result = status::failed;
          else if (out.latest->producer_submitted) {
            out.result = submission_status(*out.latest, native, owner.device_identity, out.progress);
            // Already used by a prior presentation: readiness does not grant
            // permission to reuse an old sequence for the next presentation.
            if (out.result == status::ready) {
              out.result = status::submitted;
              out.selection = selection_reason::current_already_consumed;
            }
          }
          else out.result = status::recorded;
        }
        return out;
      }
      if (!sunshine_freshness::fresh_lt(out.value->metadata.tick, now, sunshine_freshness::source_age_ms) &&
          !frame_presenting(presented_by(*out.value))) {
        out.latest = out.value; out.value = nullptr; out.result = status::stale;
        return out;
      }
      out.result = submission_status(*out.value, native, owner.device_identity, out.progress);
      out.selection = selection_reason::current_capture;
      return out;
    }
    bool pending_frame(const capture_pick &chosen, const queue_state &owner) {
      if (chosen.pending_nomination) return true;
      const auto *value = chosen.latest;
      return value && !value->nomination_only &&
        (chosen.result == status::submitted || chosen.result == status::recorded) &&
        value->metadata.frame_generation_input && (!value->finished || value->success) &&
        !value->invalid && !value->nomination_invalid && owner.provider_established &&
        owner.provider == value->metadata.provider && owner.source_id == value->metadata.source_id &&
        owner.last_epoch == value->metadata.epoch && value->metadata.sequence > owner.last_sequence;
    }
    acquisition_decision classify_acquisition(const capture_pick &chosen, const queue_state &owner,
        selection_policy policy) {
      acquisition_decision out;
      out.source_selected = chosen.value != nullptr;
      out.pending_frame = pending_frame(chosen, owner);
      const auto *value = chosen.value ? chosen.value : chosen.latest;
      if (value) {
        const auto &metadata = value->metadata;
        out.provider = metadata.provider; out.epoch = metadata.epoch;
        out.sequence = metadata.sequence; out.source_id = metadata.source_id;
        out.viewport = metadata.viewport; out.capture_id = value->id;
        out.source_valid = !value->invalid && value->failure == capture_failure::none &&
          (!value->finished || value->success);
        out.repeated_frame = !chosen.value && chosen.result != status::failed && chosen.result != status::stale &&
          chosen.result != status::ambiguous && value->finished && value->success &&
          !value->nomination_invalid && !value->invalid &&
          owner.last_epoch == metadata.epoch && owner.last_sequence == metadata.sequence;
      }
      else if (chosen.pending_nomination) {
        const auto &pending = *chosen.pending_nomination;
        out.provider = policy.require_frame_generation ? provider_kind::streamline : owner.provider;
        out.epoch = pending.epoch; out.sequence = pending.sequence;
        out.source_id = pending.source_id; out.viewport = pending.viewport;
        out.source_valid = true; // The selector already validated this bounded nomination.
      }
      // A head without a capture (refused before its slot existed), also
      // beside ambiguous or inactive views: its view names the source, so the
      // display holds only a copy of that source or its FG/SR lineage and
      // drops any other (source_changed). Until 10-28 such a pick named none
      // (epoch zero), and the display held a copy of the previous viewport or
      // NGX feature generation for up to the source age.
      else if (chosen.named) {
        const auto &head = *chosen.named;
        out.provider = chosen.provider; out.epoch = head.epoch; out.sequence = head.sequence;
        out.source_id = head.source_id; out.viewport = head.viewport;
      }
      else if (policy.require_frame_generation) {
        out.epoch = policy.epoch; out.viewport = policy.viewport;
        out.source_id = sunshine_scene_depth::encode_source_id(sunshine_scene_depth::fg_depth_key(policy.viewport));
      }
      return out;
    }
    // The selection's outcome as depth_choice records it. Describes, never decides.
    depth_choice describe_choice(const capture_pick &chosen, const acquisition_decision &decision,
        const queue_state &owner, selection_policy policy, std::uint64_t present, std::uint64_t tick, bool pixel_ready) {
      depth_choice out;
      out.present = present; out.tick_ms = tick;
      out.decision = decision;
      out.result = chosen.result; out.selection = chosen.selection;
      const auto *source = chosen.value ? chosen.value : chosen.latest;
      if (source) out.failure = source->failure;
      const bool fg = source ? source->metadata.frame_generation_input : policy.require_frame_generation;
      // A head refused before its capture existed names its source too
      // (classify_acquisition, capture_pick::named).
      if (source || chosen.pending_nomination || chosen.named || policy.require_frame_generation)
        out.rung = fg ? sunshine_game3d::source_slot::fg_depth : sunshine_game3d::source_slot::scene_depth;
      out.pixel_ready = pixel_ready;
      out.fg_required = policy.require_frame_generation; out.fg_excluded = policy.exclude_unconfirmed_fg;
      out.owner_established = owner.provider_established;
      out.owner = owner.provider; out.owner_source = owner.source_id;
      return out;
    }
    // Keep one completed API snapshot while the CPU records its successor.
    // Completion and nomination advance independently for SR as well as FG.
    // Requiring the latest nomination to finish can starve a pipelined game.
    // Finished pixels stay valid through later metadata events (camera reset,
    // tag changes, observation loss); only source identity and age bound them.
    // consumer: the reading queue. A snapshot already submitted on it is
    // ordered before the read by queue order, exactly as a current capture is
    // (submission_status); another queue needs its copy-out or a retired
    // recording (immutable_for_other_queues) and a completed producer fence.
    slot *completed_snapshot(const input &current, std::uint64_t now, std::uint64_t consumer = 0) {
      if (!valid_provider(current.provider) || !current.epoch) return nullptr;
      slot *best = nullptr;
      for (auto &value : slots) {
        const auto &metadata = value.metadata;
        if (!value.id || !value.texture || value.nomination_only || value.preservation_only ||
            !value.source_nominated || !value.finished || !value.success || value.invalid || value.nomination_invalid ||
            !value.producer_submitted ||
            metadata.provider != current.provider || metadata.frame_generation_input != current.frame_generation_input ||
            metadata.epoch != current.epoch || metadata.source_id != current.source_id || metadata.viewport != current.viewport ||
            metadata.sequence > current.sequence ||
            !(within_source_age(metadata.tick, now) || frame_presenting(presented_by(value)))) continue;
        if (!consumer || value.queue != consumer) {
          if (!immutable_for_other_queues(value)) continue;
          if (!readable_elsewhere(value, consumer, producer_progress(value))) continue;
        }
        if (!best || metadata.sequence > best->metadata.sequence ||
            (metadata.sequence == best->metadata.sequence && value.id > best->id)) best = &value;
      }
      return best;
    }
    // Two snapshots of one logical source read the same way: one storage
    // family through one view, of one resource kind. The allocation size and
    // the tag's active rectangle are not part of it: dynamic resolution gives
    // the same source a new allocation or extent every few frames (Assassin's
    // Creed Shadows with FG, 10-23: 2206x1240, 2442x1374, 2658x1494), and each
    // snapshot carries its own allocation and rectangle to its consumer.
    bool same_depth_storage(const input &a, const input &b) {
      if (!a.source || !b.source) return false;
      const auto left = a.source->desc.Format, right = b.source->desc.Format;
      return typeless(left) == typeless(right) && depth_view(left, a) == depth_view(right, b) &&
        a.resource.kind == b.resource.kind;
    }
    // head: the newest nomination the snapshot would stand in for. Its
    // storage is compared when it has a capture slot (compare_storage); a
    // head refused before its slot existed names no storage.
    selection_reason completed_fallback_reason(const queue_state &owner, std::uint64_t present,
        const input &head, const slot &completed, bool compare_storage) {
      if (completed.id <= owner.admission_after) return selection_reason::completed_before_gap;
      if (completed.metadata.sequence >= head.sequence) return selection_reason::completed_not_older;
      if (compare_storage && !same_depth_storage(completed.metadata, head)) return selection_reason::format_changed;
      // The consumed watermark belongs to the owner's provider; sequences of
      // another provider's namespace are not comparable with it.
      if (owner.provider == completed.metadata.provider && owner.last_epoch == completed.metadata.epoch &&
          completed.metadata.sequence <= owner.last_sequence)
        return selection_reason::completed_already_consumed;
      if (owner.present == present && owner.present_capture && owner.present_capture != completed.id)
        return selection_reason::presentation_already_selected;
      return selection_reason::completed_snapshot;
    }
    // A viewport's Streamline FG input (head) is pending, or a missing frame
    // (capture_pick::missing_frame: a refused DLSS-G attempt), and none of its
    // snapshots completed: FG On's first frames, or DLSS-G's compute copy
    // still in flight at its own Present in a GPU-bound game. The viewport's
    // ordinary SR/RR source of the same epoch is one display lineage with it
    // (depth_cache_update.h, lineage_successor), and its newest snapshot that
    // this queue can read now and has not consumed (the queue's watermark,
    // shared by the viewport's sources) stands in, within the source age: at
    // FG On that is the frame's own SR copy. Until 10-27 only the display's
    // retained copy was held there (FG_copy_pending), older by the frames
    // without depth before FG On, and mono once that passed the source age
    // while the frame's own SR copy was readable (the session tests' review
    // of seed c28965dca8d7c76c). The FG input takes the viewport at its first
    // readable frame, as before; a silent scope's held FG copy is still held
    // over it (frame_generation_holds), and a repeated effect of one
    // presentation keeps its one tuple.
    // A finished, valid SR/RR snapshot of the FG input fg's display lineage:
    // the viewport's ordinary Streamline source of the same epoch, no newer
    // than fg, within the source age or presented by this Present. Its
    // admission and readability are the caller's (lineage_snapshot,
    // lineage_fallback).
    bool lineage_candidate(const slot &value, const input &fg, std::uint64_t now) {
      const auto &metadata = value.metadata;
      return value.id && value.texture && !value.nomination_only && !value.preservation_only &&
        value.source_nominated && value.finished && value.success && !value.invalid && !value.nomination_invalid &&
        value.producer_submitted && metadata.provider == provider_kind::streamline && !metadata.frame_generation_input &&
        metadata.epoch == fg.epoch && metadata.viewport == fg.viewport && metadata.sequence <= fg.sequence &&
        (within_source_age(metadata.tick, now) || frame_presenting(presented_by(value)));
    }
    bool lineage_head(const input &fg) {
      return fg.provider == provider_kind::streamline && fg.frame_generation_input && fg.epoch;
    }
    slot *lineage_snapshot(const queue_state &owner, std::uint64_t native, std::uint64_t present, const input &head,
        std::uint64_t now) {
      const auto &fg = head;
      if (!lineage_head(fg) || (owner.present == present && owner.present_capture)) return nullptr;
      slot *best = nullptr;
      for (auto &value : slots) {
        const auto &metadata = value.metadata;
        if (!lineage_candidate(value, fg, now) || value.id <= owner.admission_after) continue;
        if (owner.provider == provider_kind::streamline && owner.last_epoch == metadata.epoch &&
            metadata.sequence <= owner.last_sequence) continue;
        if (best && (metadata.sequence < best->metadata.sequence ||
            (metadata.sequence == best->metadata.sequence && value.id < best->id))) continue;
        const auto progress = value.queue == native ? queue_progress{} : producer_progress(value);
        if (submission_status(value, native, owner.device_identity, progress) != status::ready) continue;
        best = &value;
      }
      return best;
    }
    // The capture pool's view of lineage_snapshot, with no queue's state
    // (completed_snapshot's consumer 0): the newest lineage candidate of the
    // FG input fg that any queue can read once its work completed. Every
    // stand-in lineage_snapshot can still pick is this one: a newer candidate
    // is unfinished GPU work or an open recording, which is never
    // reclaimable, and an older one is consumed, or below the gap, wherever
    // this one is.
    const slot *lineage_fallback(const input &fg, std::uint64_t now) {
      if (!lineage_head(fg)) return nullptr;
      const slot *best = nullptr;
      for (const auto &value : slots) {
        if (!lineage_candidate(value, fg, now) || !immutable_for_other_queues(value) ||
            !readable_elsewhere(value, 0, producer_progress(value))) continue;
        if (!best || value.metadata.sequence > best->metadata.sequence ||
            (value.metadata.sequence == best->metadata.sequence && value.id > best->id)) best = &value;
      }
      return best;
    }
    // A head refused before its capture slot existed: the identity its view
    // names. A Streamline view of a viewport's FG key is its FG input.
    input head_identity(const evaluation_head &head, provider_kind provider) {
      input out;
      out.provider = provider; out.epoch = head.epoch; out.sequence = head.sequence;
      out.source_id = head.source_id; out.viewport = head.viewport; out.tick = head.tick;
      out.frame_generation_input = provider == provider_kind::streamline &&
        head.source_id == sunshine_scene_depth::encode_source_id(sunshine_scene_depth::fg_depth_key(head.viewport));
      return out;
    }
    capture_pick select_capture(const queue_state &owner, std::uint64_t native,
        std::uint64_t present, provider_kind provider, std::uint64_t now) {
      auto chosen = select_latest_capture(owner, native, present, provider, now);
      const auto *latest = chosen.latest;
      const bool live = latest && (within_source_age(latest->metadata.tick, now) || frame_presenting(presented_by(*latest)));
      // A live source whose newest snapshot is pending may use its own newest
      // completed one; completed_snapshot matches its identity. Ownership does
      // not gate this: a pipelined source is otherwise never readable, so it
      // could never (re)establish itself after a reset, a source change or
      // another provider's expired capture.
      const bool pending = live && (chosen.result == status::recorded || chosen.result == status::submitted) &&
        !latest->invalid && !latest->nomination_invalid && !latest->nomination_only &&
        !(latest->finished && !latest->success);
      // A missing frame takes the same path: the one active view's fresh head
      // failed, was refused (before its capture slot existed too), is invalid,
      // or names its source without pixels this queue can read (a nomination-
      // only slot's pixel failure). That is no evidence against the finished
      // pixels of its display lineage, so its lineage's newest legal copy
      // stands in under the pending path's checks (identity, finished,
      // readable here, within the source age, unconsumed), and it moves no
      // admission watermark (interrupts_admission). Until 10-28 it revoked
      // depth: acquire's watermark made the newer copy completed_before_gap
      // while the display held an older one, and the Present went mono once
      // that passed the source age with a readable copy (Assassin's Creed
      // Shadows' refused SR inputs, 10-25). FG Off is one too: retire_source
      // passes the retired FG role's newest nomination to the viewport's
      // ordinary source as such a head, so the lineage's newest SR copy
      // stands in at once (10-28). Ambiguous or inactive views, a head past
      // the source age that this Present does not present, the FG scope's
      // policy picks (select_provider), an NGX feature's release
      // (retire_source) and expire_association still revoke.
      chosen.missing_frame = !pending && chosen.head && !chosen.pending_nomination && (!latest || live) &&
        chosen.result != status::ready && chosen.result != status::stale && chosen.result != status::ambiguous;
      // A nomination in progress (pending_nomination: the owner's exact
      // source, newer than its last shown frame) takes the same path too: it
      // ends recorded (the pending path above) or refused (a missing frame),
      // and either picks this same copy, so a Present meanwhile shows it now
      // rather than one Present late. It keeps its pending_nomination, which
      // still names the owner's live source (select_provider), and moves no
      // watermark. Until 10-28 it held the retained copy, one frame behind
      // the readable copy (Assassin's Creed Shadows' FG Presents racing its SR
      // evaluates), because a refusal then moved acquire's gap watermark.
      // Without a stand-in it keeps its pending result.
      const bool in_progress = chosen.pending_nomination != nullptr;
      if (!pending && !chosen.missing_frame && !in_progress) return chosen;
      const auto identity = latest ? latest->metadata :
        head_identity(in_progress ? *chosen.pending_nomination : *chosen.head, provider);
      // A missing frame or a nomination in progress without a stand-in keeps
      // its own result and reason.
      auto *completed = completed_snapshot(identity, now, native);
      // The pending head's own copy completed between select_latest_capture's
      // look and this one (each reads its producer fence anew): the frame's
      // newest copy is readable now, so it is decided again, as the current
      // capture. Until 10-28 it read completed_not_older here and the Present
      // held an older copy beside it: at FG Off, the FG copy beside the
      // frame's own readable tag copy, while the frame's SR copy on the
      // game's queue completed during that Present's selection (the 10-28
      // review's J2, once in a few fixed-set runs).
      if (pending && completed && completed == latest) {
        auto again = select_latest_capture(owner, native, present, provider, now);
        if (again.value == completed && again.result == status::ready) return again;
      }
      if (!completed) {
        if (pending) chosen.selection = selection_reason::no_completed_snapshot;
        if (auto *standin = lineage_snapshot(owner, native, present, identity, now)) {
          chosen.value = standin;
          chosen.progress = standin->queue == native ? queue_progress{} : producer_progress(*standin);
          chosen.result = status::ready;
          chosen.selection = selection_reason::lineage_snapshot;
        }
        return chosen;
      }
      // An FG head whose own completed copy is refused (published before a
      // gap, consumed, of another storage) or not readable here: its
      // lineage's newest legal SR copy stands in, as where it has none
      // (10-28; until then the Present held the display's older copy beside
      // that readable SR copy: the extra shard's seed d0dd7845ac141ad2 at
      // frame 113, under the lineage judge).
      const auto lineage_stand_in = [&] {
        auto *standin = lineage_snapshot(owner, native, present, identity, now);
        if (!standin) return false;
        chosen.value = standin;
        chosen.progress = standin->queue == native ? queue_progress{} : producer_progress(*standin);
        chosen.result = status::ready;
        chosen.selection = selection_reason::lineage_snapshot;
        return true;
      };
      const auto reason = completed_fallback_reason(owner, present, identity, *completed, latest != nullptr);
      if (reason != selection_reason::completed_snapshot) {
        if (pending) chosen.selection = reason;
        if (lineage_stand_in()) return chosen;
        if (pending && reason == selection_reason::completed_already_consumed && completed->metadata.sequence == owner.last_sequence)
          chosen.consumed_completed_capture = completed->id;
        return chosen;
      }
      const auto progress = producer_progress(*completed);
      if (submission_status(*completed, native, owner.device_identity, progress) != status::ready) {
        if (pending) chosen.selection = selection_reason::completed_not_readable;
        lineage_stand_in();
        return chosen;
      }
      // An FG copy of a frame before FG Off retired its role (its sequence
      // at or below the retiring options call's serial), recorded after the
      // Off by a race past the retirement (frame_generation_retired), then
      // FG On within the source age: the newer frame of the lineage shows,
      // whichever path names it (until 10-28 that copy stood in for FG On's
      // pending head before the lineage's newer readable SR copy of a later
      // frame was tried, the 10-28 review's P5 and J8). Any other completed
      // FG copy keeps its precedence over the SR stand-in, as the scope's
      // copies own their viewport (a live compute scope's copy in flight at
      // 4K holds the scope's previous copy, not SR).
      const auto *retired = completed->metadata.frame_generation_input ?
        fg_retirement_of(completed->metadata.epoch, completed->metadata.viewport) : nullptr;
      if (auto *standin = retired && completed->metadata.sequence <= retired->serial ?
            lineage_snapshot(owner, native, present, identity, now) : nullptr;
          standin && standin->metadata.epoch == completed->metadata.epoch &&
          standin->metadata.sequence > completed->metadata.sequence) {
        chosen.value = standin;
        chosen.progress = standin->queue == native ? queue_progress{} : producer_progress(*standin);
        chosen.result = status::ready;
        chosen.selection = selection_reason::lineage_snapshot;
        return chosen;
      }
      chosen.value = completed;
      chosen.progress = progress;
      chosen.result = status::ready;
      chosen.selection = selection_reason::completed_snapshot;
      return chosen;
    }
    // A genuine source interruption revokes the queue's snapshots from before
    // it (admission_after). A pending, repeated or missing frame
    // (capture_pick::missing_frame; FG Off's retired FG head too, which
    // retire_source passes to the viewport's ordinary source) does not, nor
    // does a newest copy that this Present still presents
    // (frame_presenting): its frame's copies stay admissible.
    bool interrupts_admission(const queue_state &owner, const capture_pick &chosen) {
      return owner.provider_established && !chosen.missing_frame &&
        chosen.result != status::ready && chosen.result != status::recorded && chosen.result != status::submitted &&
        !(chosen.result == status::stale && chosen.latest && frame_presenting(presented_by(*chosen.latest)));
    }
    capture_pick select_provider(const queue_state &owner, std::uint64_t native,
        std::uint64_t present, std::uint64_t now, selection_policy policy = {}) {
      presenting_generated = policy.generated_presents;
      const auto fg_head = [](const evaluation_head *head) {
        return head && head->source_id == sunshine_scene_depth::encode_source_id(sunshine_scene_depth::fg_depth_key(head->viewport));
      };
      const auto select_sl = [&] {
        auto chosen = select_capture(owner, native, present, provider_kind::streamline, now);
        const auto *source = chosen.value ? chosen.value : chosen.latest;
        // Without a capture, a nomination in progress or a head refused before
        // its slot existed names the source: an FG input's is excluded as one.
        if (policy.exclude_unconfirmed_fg &&
            (source ? source->metadata.frame_generation_input :
              fg_head(chosen.pending_nomination ? chosen.pending_nomination : chosen.head))) {
          chosen = {};
          chosen.result = status::stale;
          chosen.selection = selection_reason::fg_scope_missing;
          return chosen;
        }
        return chosen;
      };
      if (policy.require_frame_generation) {
        if (!policy.epoch) { capture_pick missing; missing.selection = selection_reason::fg_scope_missing; return missing; }
        // FG owns the source choice before pixels are ready. Do not silently
        // substitute SR depth while its matching camera belongs to SL FG.
        auto fg = select_sl();
        const auto fg_source = sunshine_scene_depth::encode_source_id(sunshine_scene_depth::fg_depth_key(policy.viewport));
        const auto matches = [&](const sunshine_scene_depth::frame &value) {
          return value.provider == provider_kind::streamline && value.frame_generation_input &&
            value.epoch == policy.epoch && value.viewport == policy.viewport &&
            value.source_id == fg_source;
        };
        // The lineage's ordinary snapshot stands in for its pending input
        // (lineage_snapshot), of this scope's viewport and epoch.
        const bool standin = fg.value && fg.selection == selection_reason::lineage_snapshot &&
          fg.value->metadata.epoch == policy.epoch && fg.value->metadata.viewport == policy.viewport;
        // Without a capture, a nomination in progress or a head refused before
        // its slot existed names the source it must match.
        const auto *named = fg.latest ? nullptr : fg.pending_nomination ? fg.pending_nomination : fg.head;
        if ((fg.value && !standin && !matches(fg.value->metadata)) || (fg.latest && !matches(fg.latest->metadata)) ||
            (named && (named->epoch != policy.epoch || named->viewport != policy.viewport ||
              named->source_id != fg_source))) {
          capture_pick mismatch; mismatch.selection = selection_reason::fg_scope_mismatch; return mismatch;
        }
        return fg;
      }
      // Repeated effects in one presentation keep one immutable provider tuple.
      if (owner.provider_established && owner.present == present && owner.present_capture)
        return owner.provider == provider_kind::streamline ? select_sl() : select_capture(owner, native, present, owner.provider, now);
      auto sl = select_sl();
      auto ngx = select_capture(owner, native, present, provider_kind::ngx, now);
      // Ownership never moves to depth older than the owner's last selected
      // copy (present_capture). Capture ids follow publication order across
      // both providers, while GPU readability need not: the other provider's
      // copy of an earlier frame can turn readable after the owner's newer
      // one was shown. Such a pick is no readable source, only a live one,
      // so the owner keeps selection and the display holds its copy until
      // the other provider's next copy is readable (one frame at most). Until
      // 10-28 a failing owner handed over to it, older depth after newer, and
      // the owner's shown copy then stood in again as a missing frame's,
      // flipping ownership back (the consumed watermark restarts at a
      // handover and cannot compare the two namespaces).
      const auto older_than_shown = [&](capture_pick &value, provider_kind provider) {
        if (!owner.provider_established || owner.provider == provider || !owner.present_capture || !value.value ||
            value.value->id > owner.present_capture) return;
        value.value = nullptr;
        value.result = status::recorded;
        value.selection = selection_reason::completed_older_than_shown;
      };
      older_than_shown(sl, provider_kind::streamline);
      older_than_shown(ngx, provider_kind::ngx);
      const auto readable = [](const capture_pick &value) { return value.value && value.result == status::ready; };
      // Expired: no active view, or a newest nomination older than the source
      // age. A pick without any slot (an attempt rejected before its capture
      // existed) is not expired while its view is active.
      const auto expired = [&](const capture_pick &value) {
        return value.result == status::stale || (value.latest && !within_source_age(value.latest->metadata.tick, now) &&
          !frame_presenting(presented_by(*value.latest)));
      };
      // A source is live while its newest nomination is within the source age.
      // Readable implies live; liveness never extends that age.
      const auto live = [&](const capture_pick &value) {
        return value.pending_nomination || (value.latest && !expired(value));
      };
      // A live source whose newest valid snapshot is still on its way, so its
      // completed snapshot can become readable. A failed or refused head
      // (capture_pick::missing_frame) moves no gap watermark either, but has
      // no pixels of its own to wait for: its lineage's completed copy, when
      // one is legal, already stood in (select_capture).
      const auto pending = [&](const capture_pick &value) {
        const auto *latest = value.latest;
        return value.pending_nomination || (live(value) && !latest->nomination_only && !latest->invalid &&
          !latest->nomination_invalid && (!latest->finished || latest->success) &&
          (value.result == status::recorded || value.result == status::submitted));
      };
      // Ownership only arbitrates between live sources. Only a delivering owner
      // (one frame copied within the source age) keeps its own live source:
      // while that source is readable, or while its real successor is pending,
      // so a short pipeline delay does not alternate providers every frame. Its
      // already copied frame is not a successor. This preserves selection only;
      // display-cache rules still govern old pixels. Failed, unsupported,
      // missing, replaced, expired or undelivered owner evidence never holds
      // against a readable source; a failed head's completed copy that stood
      // in (select_capture) is the owner's readable source.
      const auto protected_owner = [&](const capture_pick &value) {
        const auto *source = value.value ? value.value : value.latest;
        if (source) {
          if (source->metadata.epoch != owner.nominated_epoch || source->metadata.source_id != owner.source_id ||
              source->metadata.viewport != owner.viewport) return false;
        } else if (!value.pending_nomination) return false; // Already bound to the owner's exact source.
        if (!within_source_age(owner.last_source_tick, now)) return false;
        if (readable(value) || value.pending_nomination) return true;
        return pending(value) && (value.latest->metadata.epoch != owner.last_epoch ||
          value.latest->metadata.sequence > owner.last_sequence);
      };
      const auto *own = owner.provider_established ? owner.provider == provider_kind::streamline ? &sl : &ngx : nullptr;
      if (own && protected_owner(*own)) return *own;
      // No protected owner: readable SL is preferred over readable NGX. This
      // includes an established SL owner whose readable frames are never copied
      // (failed display preparation or copy): it is not delivering, yet keeps
      // that preference until its source is revoked or stops being readable.
      if (readable(sl)) return sl;
      if (readable(ngx)) return ngx;
      // Neither provider is readable. A delivering owner keeps any pick that
      // has not expired: after one rejected, failed or ambiguous attempt, also
      // one rejected before its capture slot existed, the display may hold its
      // last copied frame, and the other provider's pixel-less pending capture,
      // such as a nested NGX evaluation still in flight, takes no authority. A
      // persistently failing owner stops delivering within the source age and
      // so cannot lock out a pending source. Otherwise a pending live pick ranks
      // above a failing live one, whose lineage had no completed copy to stand
      // in (a failing pick no longer marks a gap, interrupts_admission, but a
      // pending one may still become readable), and any live pick ranks above
      // an expired one. An owner that has stopped
      // delivering keeps its provider only while the other provider does not
      // rank higher.
      const auto rank = [&](const capture_pick &value) { return pending(value) ? 2 : live(value) ? 1 : 0; };
      if (own) {
        if (!expired(*own) && within_source_age(owner.last_source_tick, now)) return *own;
        const auto &other = own == &sl ? ngx : sl;
        return rank(other) > rank(*own) ? other : *own;
      }
      if (rank(sl) != rank(ngx)) return rank(sl) > rank(ngx) ? sl : ngx;
      // Preserve pending/failure diagnostics when neither provider is ready.
      // Pending SL cannot block a subsequently readable NGX source.
      if (sl.value || (!ngx.value && (sl.latest || !evaluations[provider_index(provider_kind::ngx)].sequence))) return sl;
      return ngx;
    }
    void establish_provider(queue_state &owner, const slot &value, std::uint64_t present) {
      // Ownership moving to the other provider ends the previous provider's
      // presentation: every copy recorded before the new owner's (its own
      // older ones too) is historical. The consumed watermark below restarts
      // and cannot compare the two namespaces' sequences, so without this the
      // previous provider's copy already shown, or one older than the new
      // owner's, could be selected again and flip ownership back (a missing
      // frame no longer revokes them, select_capture). Copies recorded after
      // it, of either provider, stay admissible. The previous owner's last
      // selected copy is historical however the new one was chosen (also a
      // required FG scope's pick, which select_provider's older_than_shown
      // does not judge). So is it after FG Off retired the owner's FG role
      // (retire_source keeps its present_capture): an NGX copy taking the
      // queue then makes the viewport's SR copies before it historical, as
      // any handover does, so FG On's lineage stand-in never steps back to
      // one of them.
      if ((owner.provider_established || owner.present_capture) && owner.provider != value.metadata.provider && value.id)
        owner.admission_after = std::max({owner.admission_after, value.id - 1, owner.present_capture});
      if (!owner.provider_established || owner.provider != value.metadata.provider ||
          owner.source_id != value.metadata.source_id || owner.nominated_epoch != value.metadata.epoch)
        owner.last_epoch = owner.last_sequence = owner.last_source_tick = 0;
      owner.provider_established = true;
      owner.provider = value.metadata.provider;
      owner.source_id = value.metadata.source_id;
      owner.viewport = value.metadata.viewport;
      owner.present = present; owner.present_capture = value.id;
      owner.nominated_epoch = value.metadata.epoch;
    }
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
    // Requires mutex. A depth copy recorded on a COMPUTE list is its FG
    // scope's accepted copy (frame_generation_scope_live) only once the
    // Present queue can read it: its SDK call succeeded, its submission
    // carries the capture owner's fence, it was copied out on its compute
    // queue (or, without a copy-out, its recording was reset) and that fence
    // passed (completed_snapshot's rule for another queue's copy). The scope
    // then turns live with readable FG pixels, so its handover drops no SR
    // depth for a copy still in flight. Streamline resets its DLSS-G
    // recording frames later, which before copy-out (10-27) kept each such
    // copy unreadable for three of its evaluations. One never submitted, as
    // when its queue has no place, or never readable keeps no other depth out
    // of its viewport. Called at the copy's call, its submission and each
    // liveness query (frame_generation_scope_live).
    void accept_compute_copy(const slot &value) {
      const auto &m = value.metadata;
      if (!value.compute_list || !value.source_nominated || !value.finished || !value.success || value.invalid ||
          value.nomination_only || value.preservation_only || !value.producer_submitted || !value.producer_fence ||
          !valid_provider(m.provider))
        return;
      auto &views = evaluations[provider_index(m.provider)].views;
      const auto owned = [&](const auto &view) {
        return view.epoch == m.epoch && view.viewport == m.viewport && view.source_id == m.source_id;
      };
      if (std::none_of(views.begin(), views.end(), [&](const auto &view) {
            return owned(view) && view.accepted_sequence < m.sequence; }) ||
          !immutable_for_other_queues(value))
        return;
      // Readable on the queue that reads depth now (U-C1: ordered there by the
      // game's own fences, or complete).
      if (!readable_elsewhere(value, reader_queue(), producer_progress(value))) return;
      for (auto &view : views)
        if (owned(view)) {
          view.accepted_tick = std::max(view.accepted_tick, m.tick);
          view.accepted_sequence = std::max(view.accepted_sequence, m.sequence);
        }
    }
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
    // After the association's silence (freshness.h, association_ms).
    // Requires mutex.
    void expire_association(queue_state &owner, std::uint64_t now) {
      if (!owner.provider_established || !valid_provider(owner.provider)) return;
      const auto &current = evaluations[provider_index(owner.provider)];
      // The owner's own evaluations: its source and epoch, so the malformed-
      // input marker (epoch 0, source 0) never keeps a Streamline SR owner
      // (source 0) associated.
      const auto within = [&](std::uint64_t tick) {
        return tick && sunshine_freshness::fresh_le(tick, now, sunshine_freshness::association_ms);
      };
      const auto recent = [&](const evaluation_head &head) {
        return head.source_id == owner.source_id && head.epoch == owner.nominated_epoch && within(head.tick) &&
          (!head.rejected_since || within(head.rejected_since));
      };
      if (recent(current.pending_nomination) || std::any_of(current.views.begin(), current.views.end(), recent)) return;
      owner.provider_established = false;
      owner.source_id = owner.last_epoch = owner.last_sequence = owner.last_source_tick = owner.nominated_epoch = owner.present_capture = 0;
      // Snapshots from before the silence never re-establish it.
      owner.admission_after = std::max(owner.admission_after, serial.load(std::memory_order_relaxed));
    }
    // frame_generation_scope_live's clock. The seam characterization fixes it
    // to pin the scope's exact edges (streamline_depth_capture_test_access.h).
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
    std::atomic<std::uint64_t> scope_clock{};
    std::uint64_t scope_now() {
      const auto fixed = scope_clock.load(std::memory_order_relaxed);
      return fixed ? fixed : GetTickCount64();
    }
#else
    std::uint64_t scope_now() { return GetTickCount64(); }
#endif
  }
  bool provider_active(std::uint64_t native) {
    if (!requested.load() || !native) return false;
    std::lock_guard lock(mutex);
    auto *owner = queue(native);
    if (!requested.load() || !owner || owner->retiring) return false;
    expire_association(*owner, GetTickCount64());
    return owner->provider_established;
  }
  bool evaluation_live(std::uint64_t now_ms, std::uint64_t window_ms) {
    if (!requested.load()) return false;
    std::lock_guard lock(mutex);
    // Any real evaluation: Streamline SR uses source 0 by design, and only the
    // malformed-input marker begins one with epoch 0.
    const auto recent = [&](const evaluation_head &head) {
      return head.epoch && head.tick && sunshine_freshness::fresh_le(head.tick, now_ms, window_ms);
    };
    for (const auto &space : evaluations) {
      if (recent(space.pending_nomination)) return true;
      for (const auto &view : space.views) if (recent(view)) return true;
    }
    return false;
  }
  SUNSHINE_CAPTURE_INTERNAL {
    // Requires mutex. The FG scope's view when the scope is live
    // (frame_generation_scope_live), else null.
    evaluation_head *live_frame_generation_scope(std::uint64_t epoch, std::uint32_t viewport,
        std::uint64_t enabled_tick) {
      const auto now = scope_now();
      const auto source = sunshine_scene_depth::encode_source_id(sunshine_scene_depth::fg_depth_key(viewport));
      // A compute-list copy of this scope counts once readable (accept_compute_copy).
      if (compute_copies_recorded)
        for (const auto &value : slots)
          if (value.id && value.compute_list && value.metadata.epoch == epoch && value.metadata.viewport == viewport &&
              value.metadata.source_id == source)
            accept_compute_copy(value);
      // A tick slightly ahead of this read (another thread's) is fresh.
      for (auto &view : evaluations[provider_index(provider_kind::streamline)].views)
        if (view.epoch == epoch && view.viewport == viewport && view.source_id == source && view.accepted_tick &&
            view.accepted_tick >= enabled_tick && view.accepted_sequence > view.missed_sequence &&
            sunshine_freshness::ahead_tolerant_le(view.accepted_tick, now, sunshine_freshness::fg_silent_ms))
          return &view;
      return nullptr;
    }
    // Requires mutex. value, a copy just recorded, supplies its FG scope's
    // current frame (frame_generation_frame).
    void record_frame_generation_copy(const input &value) {
      if (!value.frame_generation_input || !valid_provider(value.provider)) return;
      for (auto &view : evaluations[provider_index(value.provider)].views)
        if (view.epoch == value.epoch && view.viewport == value.viewport && view.source_id == value.source_id) {
          view.recorded_sequence = std::max(view.recorded_sequence, value.sequence);
          view.awaiting = false;
        }
    }
  }
  bool frame_generation_scope_live(std::uint64_t epoch, std::uint32_t viewport, std::uint64_t enabled_tick) {
    if (!requested.load() || !epoch) return false;
    std::lock_guard lock(mutex);
    return live_frame_generation_scope(epoch, viewport, enabled_tick) != nullptr;
  }
  void frame_generation_enabled(std::uint64_t enabled_tick) {
    frame_generation_on_moves.store(reader_moves.load(std::memory_order_relaxed), std::memory_order_relaxed);
    frame_generation_on_tick.store(enabled_tick, std::memory_order_relaxed);
    if (!enabled_tick) frame_generation_run_tick.store(0, std::memory_order_relaxed);
  }
  void frame_generation_activity(std::uint64_t now) {
    const auto last = frame_generation_activity_tick.exchange(now, std::memory_order_relaxed);
    const auto on = frame_generation_on_tick.load(std::memory_order_relaxed);
    if (!last || now < last || now - last >= sunshine_freshness::fg_silent_ms || (on && on >= last)) {
      frame_generation_run_moves.store(reader_moves.load(std::memory_order_relaxed), std::memory_order_relaxed);
      frame_generation_run_tick.store(now, std::memory_order_relaxed);
    }
  }
  void frame_generation_frame(std::uint64_t epoch, std::uint32_t viewport, std::uint64_t enabled_tick) {
    if (!requested.load() || !epoch) return;
    frame_generation_activity(GetTickCount64());
    std::lock_guard lock(mutex);
    auto *view = live_frame_generation_scope(epoch, viewport, enabled_tick);
    if (!view || !std::exchange(view->awaiting, true)) return;
    // No copy recorded since the previous frame began: that frame had no FG
    // copy. Every copy recorded so far, accepted or still in flight, is of
    // an earlier frame (frame_generation_missed's ordering).
    view->missed_sequence = std::max({view->missed_sequence, view->recorded_sequence, view->accepted_sequence});
    view->awaiting = false;
  }
  void frame_generation_missed(std::uint64_t epoch, std::uint32_t viewport, std::uint64_t sequence) {
    if (!requested.load() || !epoch || !sequence) return;
    std::lock_guard lock(mutex);
    const auto source = sunshine_scene_depth::encode_source_id(sunshine_scene_depth::fg_depth_key(viewport));
    for (auto &view : evaluations[provider_index(provider_kind::streamline)].views)
      if (view.epoch == epoch && view.viewport == viewport && view.source_id == source)
        view.missed_sequence = std::max(view.missed_sequence, sequence);
  }
  bool provider_identity(std::uint64_t native, provider_kind &provider, std::uint64_t &source_id) {
    source_id = 0;
    if (!requested.load() || !native) return false;
    std::lock_guard lock(mutex);
    auto *owner = queue(native);
    if (!owner || owner->retiring) return false;
    expire_association(*owner, GetTickCount64());
    if (!owner->provider_established) return false;
    provider = owner->provider;
    source_id = owner->source_id;
    return true;
  }

  void frame_generation_restored(std::uint64_t epoch, std::uint32_t viewport) {
    if (!requested.load() || !epoch) return;
    std::lock_guard lock(mutex);
    if (auto *mark = fg_retirement_of(epoch, viewport)) mark->active = false;
  }
  bool frame_generation_retired(std::uint64_t epoch, std::uint32_t viewport, std::uint64_t serial) {
    if (!requested.load() || !epoch) return false;
    std::lock_guard lock(mutex);
    const auto *mark = fg_retirement_of(epoch, viewport);
    return mark && (mark->active || (serial && serial <= mark->serial));
  }


  std::uint64_t game_executions() { return game_execution_count.load(std::memory_order_acquire); }
  std::uint64_t next_present_ordinal() { return present_ordinal(); }
  void evaluation_called(std::uint64_t epoch) {
    if (!requested.load() || !epoch) return;
    std::lock_guard lock(mutex);
    newest_evaluation_present = std::max(newest_evaluation_present, present_ordinal());
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
  void present(std::uint64_t native) {
    // Counted for every Present, also one the capture owner does not serve
    // (frame_presenting must never take a frame older than the last Present
    // for the current one).
    presents.fetch_add(1, std::memory_order_acq_rel);
    if (!requested.load() || !native) return;
    std::lock_guard lock(mutex);
    const auto *owner = queue(native);
    if (!owner) return;
    const auto device = device_cookie(owner->device.p);
    if (!device) return;
    for (const auto &weak : sources) if (const auto source = weak.lock())
      if (source->device_identity == device) ++source->present_generation;
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

  bool acquire(std::uint64_t native, std::uint64_t present, packet &out, capture_diagnostic *diagnostic,
      selection_policy policy) {
    acquisition_decision ignored;
    return acquire(native, present, out, ignored, diagnostic, policy);
  }
  bool acquire(std::uint64_t native, std::uint64_t present, packet &out, acquisition_decision &decision,
      capture_diagnostic *diagnostic, selection_policy policy) {
    out = {};
    decision = {};
    if (policy.require_frame_generation) {
      decision.epoch = policy.epoch; decision.viewport = policy.viewport;
      decision.source_id = sunshine_scene_depth::encode_source_id(sunshine_scene_depth::fg_depth_key(policy.viewport));
    }
    if (diagnostic) {
      *diagnostic = {}; diagnostic->requested_queue = native;
    }
    queue_progress progress;
    bool recording_retired = false;
    const auto result = [diagnostic, &decision, &progress, &recording_retired, &policy](status why,
        const slot *value = nullptr) {
      if (diagnostic) {
        diagnostic->result = why;
        diagnostic->frame_generation_input = value ? value->metadata.frame_generation_input :
          policy.require_frame_generation;
        diagnostic->capture_id = decision.capture_id; diagnostic->sequence = decision.sequence;
        diagnostic->epoch = decision.epoch; diagnostic->source_id = decision.source_id;
        diagnostic->viewport = decision.viewport; diagnostic->provider = decision.provider;
        diagnostic->repeated_frame = decision.repeated_frame; diagnostic->pending_frame = decision.pending_frame;
        if (value) {
          diagnostic->failure = value->failure;
          diagnostic->source_tick_ms = value->metadata.tick;
          diagnostic->command = value->command;
          diagnostic->queue = value->queue; diagnostic->producer_fence = value->producer_fence;
          diagnostic->producer_completed = progress.completed;
          diagnostic->producer_completion_valid = progress.valid();
          diagnostic->producer_recording_retired = recording_retired;
          for (const auto &point : value->retirements)
            if (point.queue == diagnostic->consumer_queue) diagnostic->retire_fence = point.value;
          diagnostic->finished = value->finished; diagnostic->success = value->success;
          diagnostic->submitted = value->producer_submitted; diagnostic->invalid = value->invalid;
        }
      }
      reason = why;
      return why == status::ready;
    };
    if (!requested.load() || !native || !native_observer::queue_ready(native)) return result(status::inactive);
    std::lock_guard lock(mutex);
    auto *owner = queue(native);
    if (!owner || owner->retiring) return result(status::unsupported_queue);
    native = reinterpret_cast<std::uint64_t>(owner->queue.p);
    if (diagnostic) diagnostic->consumer_queue = native;
    const auto selection_tick = GetTickCount64();
    const auto chosen = select_provider(*owner, native, present, selection_tick, policy);
    progress = chosen.progress;
    const auto *selected = chosen.value ? chosen.value : chosen.latest;
    if (selected) {
      // Diagnostic completion facts only; never changes admission or GPU ordering.
      if (selected->queue == native) progress = producer_progress(*selected);
      recording_retired = producer_recording_retired(*selected);
    }
    decision = classify_acquisition(chosen, *owner, policy);
    if (diagnostic) {
      diagnostic->selection = chosen.selection;
      diagnostic->consumed_completed_capture = chosen.consumed_completed_capture;
      diagnostic->selection_tick_ms = selection_tick;
      if (selected || policy.require_frame_generation || owner->provider_established) {
        const auto provider = selected ? selected->metadata.provider :
          policy.require_frame_generation ? provider_kind::streamline : owner->provider;
        const auto &current = evaluations[provider_index(provider)];
        for (const auto &view : current.views) if (view.epoch == current.epoch && view.sequence) {
          diagnostic->newest_view_tick_ms = std::max(diagnostic->newest_view_tick_ms, view.tick);
          // As select_latest_capture counts them (frame_presenting too).
          if (sunshine_freshness::fresh_lt(view.tick, selection_tick, sunshine_freshness::source_age_ms) ||
              view_presenting(view, provider))
            ++diagnostic->active_view_count;
        }
      }
    }
    // A genuine source interruption revokes snapshots from before the gap;
    // pending, repeated and missing frames do not (interrupts_admission).
    if (interrupts_admission(*owner, chosen)) owner->admission_after = serial.load(std::memory_order_relaxed);
    // The display may hold the newest evaluated frame's own copy at this
    // Present, whatever its age (depth_cache_update.h, own_frame).
    decision.presenting_floor = frame_presenting(newest_evaluation_present) ? newest_evaluation_present : 0;
    // A head refused before its capture existed names its sequence too, so a
    // missing frame's stand-in reads as sequence < newest_sequence.
    if (diagnostic) diagnostic->newest_sequence = chosen.latest ? chosen.latest->metadata.sequence :
      chosen.pending_nomination ? chosen.pending_nomination->sequence : chosen.head ? chosen.head->sequence : 0;
    if (diagnostic && chosen.pending_nomination) {
      const auto &pending = *chosen.pending_nomination;
      diagnostic->source_tick_ms = pending.tick;
    }
    if (!chosen.value) {
      owner->choice = describe_choice(chosen, decision, *owner, policy, present, selection_tick, false);
      owner->chosen = true; read_on(owner->cookie, false);
      return result(chosen.result, chosen.latest);
    }
    auto *best = chosen.value;
    // Borrowed CPU ownership alone does not imply a GPU consumer. Only a
    // successful pre-read mark_consumer establishes that retirement obligation.
    bind_pixel_consumer(*best, native, chosen.result);
    establish_provider(*owner, *best, present);
    describe_packet(*best, native, out);
    out.pixel_ready = out.pixel_ready && chosen.result == status::ready;
    owner->choice = describe_choice(chosen, decision, *owner, policy, present, selection_tick, out.pixel_ready);
    owner->chosen = true; read_on(owner->cookie, out.pixel_ready);
    result(chosen.result, best);
    return true; // Source authority and current pixels are separate outcomes.
  }
  bool choice(std::uint64_t native, depth_choice &out) {
    out = {};
    if (!native) return false;
    std::lock_guard lock(mutex);
    const auto *owner = queue(native);
    if (!owner || !owner->chosen) return false;
    out = owner->choice;
    return true;
  }
  void complete_frame(const packet &value, std::uint64_t present) {
    std::lock_guard lock(mutex);
    auto *owner = queue(value.queue);
    // The presentation this Present selected and its consumer copied. FG Off
    // on the game's thread between that selection and this copy (DLSS-G
    // presenting on its own) retires the owner's FG role but keeps that
    // presentation tuple (retire_source), so the copied frame is still the
    // consumed one: until 10-28 it was not, and an SR copy of an earlier
    // frame published after it could show at the next Present, older depth
    // after newer (the 10-28 review's P4). Every other end of an owner clears
    // present_capture.
    if (!owner || owner->present != present || owner->present_capture != value.capture_id ||
        owner->provider != value.metadata.provider ||
        (owner->provider_established && owner->nominated_epoch != value.metadata.epoch)) return;
    owner->last_epoch = value.metadata.epoch;
    owner->last_sequence = value.metadata.sequence;
    owner->last_source_tick = value.metadata.tick;
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
#include "depth_capture/record.inl"
#include "depth_capture/observation.inl"
#include "depth_capture/queues.inl"
#include "depth_capture/queue_order_consumer.inl"
#include "depth_capture/pool.inl"
