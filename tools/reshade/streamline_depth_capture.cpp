// SPDX-License-Identifier: GPL-3.0-only
#include "streamline_depth_capture.h"
#include "capture_state_policy.h"
#include "streamline_native_observer.h"
#include "native_command_storage.h"
#include "native_resource_identity.h"
#include "depth_view_format.h"
#include "freshness.h"
#include "streamline_depth_capture_test_access.h"
#include "../../src/game3d_debug_formats.h"
#include <d3d12.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
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

    unsigned provider_index(provider_kind value) { return static_cast<unsigned>(value); }
    bool valid_provider(provider_kind value) { return provider_index(value) < provider_count; }
    void set_attempt(provider_kind provider, status value) {
      reason = value;
      if (valid_provider(provider)) attempt_status[provider_index(provider)] = value;
    }
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
    // When FG options last turned on (frame_generation_enabled), 0 never.
    // FG moves Present to DLSS-G's queue at its first Present, after the
    // render queue's copies of that frame were recorded and submitted while
    // that queue still read depth, so for the source age after FG turned on
    // the reader's own copies are copied out too (U-C1b,
    // reader_may_move).
    std::atomic<std::uint64_t> frame_generation_on_tick{};
    std::array<evaluation_namespace, provider_count> evaluations;

    // A nomination's identity: its evaluation's epoch, sequence, source and
    // viewport.
    evaluation_head nomination_of(const input &value) {
      return {value.epoch, value.sequence, 0, value.source_id, value.viewport};
    }
    bool same_nomination(const evaluation_head &a, const evaluation_head &b) {
      return a.epoch == b.epoch && a.sequence == b.sequence && a.source_id == b.source_id && a.viewport == b.viewport;
    }
    bool matching_evaluation(const evaluation_head &head, const input &value) {
      return same_nomination(head, nomination_of(value));
    }
    // Ends provider's nomination in progress when it is still this one.
    void end_nomination(provider_kind provider, const evaluation_head &nomination) {
      std::lock_guard lock(mutex);
      if (valid_provider(provider)) {
        auto &pending = evaluations[provider_index(provider)].pending_nomination;
        if (same_nomination(pending, nomination)) pending = {};
      }
    }
    void end_nomination(const input &value) { end_nomination(value.provider, nomination_of(value)); }

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

    command_storage::handle command(std::uint64_t native, std::uint64_t cookie, bool create = false) {
      if (!native || !cookie) return {};
      auto entry = command_storage::acquire(reinterpret_cast<ID3D12Object *>(native), create);
      if (!entry) return {};
      if (!entry->cookie && create) {
        entry->cookie = cookie; entry->observation_generation = command_generation;
        entry.life()->cookie.store(cookie, std::memory_order_release);
      }
      if (entry->cookie != cookie) return {};
      if (entry->observation_generation != command_generation) {
        if (!entry->invalid) entry->invalidation = recording_loss::global_observation_loss;
        entry->invalid = true;
      }
      return entry;
    }
    void invalidate(command_state &owner, recording_loss why);
    void associate_recording(std::uint64_t native, std::uint64_t *out_cookie) {
      if (!out_cookie) return;
      *out_cookie = 0;
      // Native method entry authenticates a live command object. A fresh list
      // is already open after CreateCommandList and need not call Reset first.
      // Never initialize from a post-call zero-cookie observation: Reset could
      // have changed its lifetime while the original operation was executing.
      std::lock_guard lock(mutex);
      if (!native || !native_observer::recording_cookie_absent(native)) return;
      auto *object = reinterpret_cast<ID3D12Object *>(native);
      if (command_storage::acquire(object, false)) return;
      auto owner = command_storage::acquire(object, true);
      if (!owner || owner->cookie) return;
      const auto cookie = ++serial;
      if (!native_observer::set_recording_cookie(native, cookie)) {
        invalidate(*owner, recording_loss::source_identity_unavailable);
        return;
      }
      owner->restart(cookie, command_generation);
      owner.life()->cookie.store(cookie, std::memory_order_release);
      *out_cookie = cookie;
    }
    // Protected by mutex: orders the queue places by their last use
    // (queue_state::last_use), so free_place reclaims the longest unused.
    std::uint64_t place_clock{};
    queue_state *known_queue(std::uint64_t native) {
      for (auto &value : queues) if (value && reinterpret_cast<std::uint64_t>(value->queue.p) == native) {
        value->last_use = ++place_clock;
        return value.get();
      }
      return nullptr;
    }
    queue_state *queue(std::uint64_t native) {
      if (auto *value = known_queue(native)) return value;
      // A COM wrapper and its native queue share private data, just like the
      // command-recording cookie. Never substitute device identity for a queue.
      com_ptr<ID3D12CommandQueue> object;
      std::uint64_t cookie{}; UINT size = sizeof(cookie);
      if (!query_native(native, IID_ID3D12CommandQueue, object) ||
          FAILED(object->GetPrivateData(queue_guid, &size, &cookie)) || size != sizeof(cookie) || !cookie) return nullptr;
      for (auto &value : queues) if (value && value->cookie == cookie) {
        value->last_use = ++place_clock;
        return value.get();
      }
      return nullptr;
    }
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
    bool fail_state_growth{};
#endif
    source_state *state(command_state &owner, std::uint64_t identity, bool create) {
      if (!identity) return nullptr;
      for (auto &value : owner.states) if (value.source == identity) return &value;
      if (create) {
        try {
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
          if (fail_state_growth && owner.states.size() == owner.states.capacity()) throw std::bad_alloc();
#endif
          if (owner.states.capacity() == 0) owner.states.reserve(16);
          owner.states.push_back({identity});
          return &owner.states.back();
        } catch (...) {
          // Hook callbacks must not throw. Existing callers invalidate the
          // recording if preserving a source's state/block actually fails.
          return nullptr;
        }
      }
      return nullptr;
    }
    void invalidate(command_state &owner, recording_loss why) {
      if (!owner.invalid) owner.invalidation = why;
      owner.invalid = true;
    }
    void block_source(command_state &owner, ID3D12Resource *resource) {
      // Exceptional barriers can precede the first tag. Retain identity on the
      // actual resource, without registering ordinary game resources globally.
      const auto identity = retain_source_cookie(resource);
      auto *value = state(owner, identity, true);
      if (!value) { invalidate(owner, identity ? recording_loss::source_state_capacity : recording_loss::source_identity_unavailable); return; }
      value->blocked = true;
    }
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
    // storage the idle sweep released unused (released).
    struct {
      std::atomic<std::uint64_t> made{}, ring_busy{}, failed{}, predicted{}, taken{}, unstored{}, released{};
    } copy_out_counters;
    // Reads admitted on another queue by the queue-order model before their
    // producer fence completed (U-C1; the queue order line's proofs).
    std::atomic<std::uint64_t> order_inflight_reads{};
    void invalidate_all() {
      std::lock_guard lock(mutex);
      for_each_capture([](auto &value) { invalidate(value, capture_failure::observer_loss); });
      ++command_generation;
      // Lists executed meanwhile may have gone unobserved.
      proven.unprove_all();
      reason = status::unavailable;
    }
    // Every game ResourceBarrier/Barrier arrives here. A recording's payload is
    // read and written only on the thread recording that list (its lifecycle
    // events, record_impl and this hook; a list is recorded by one thread at
    // a time), so barrier states take no capture lock. A list ReShade never
    // reported cannot admit a capture (lifecycle_view), so nothing is tracked
    // for it and its states cannot grow without a Reset.
    void barriers(std::uint64_t native, std::uint64_t cookie, std::uint32_t count, const D3D12_RESOURCE_BARRIER *values) {
      auto owner = command(native, cookie, true);
      if (!owner) return;
      // A closed recording records nothing: the list was Reset unreported.
      if (owner->closed) { owner->missed_reset = true; return; }
      if (!owner->reported) return;
      ++owner->native_barrier_calls;
      owner->last_barrier_command = native;
      for (unsigned n = 0; n != count; ++n) {
        const auto &value = values[n];
        // An aliasing barrier, named or wildcard (NULL: any placed or reserved
        // resource), changes no resource state; it only moves heap memory
        // between overlapping resources. Every capture here copies where the
        // SDK consumes the source (its tag call or evaluation), so the game
        // must have that source active and initialized there. Transient-memory
        // engines (RE Engine) alias every frame; blocking on them refused all
        // depth. The generic bind-switch preservation copies at no such point
        // and keeps its own alias rule (depth_addon.cpp). The end-of-frame
        // copy copies at no such point either: an aliasing barrier is marked
        // for the per-resource proven state alone, which it revokes
        // (proven_state.h). A resource without an identity cannot be tracked
        // there, so none is created for it.
        if (value.Type == D3D12_RESOURCE_BARRIER_TYPE_ALIASING) {
          const auto order = ++owner->barrier_order;
          const auto &alias = value.Aliasing;
          if (!alias.pResourceBefore && !alias.pResourceAfter) { owner->wildcard_alias = order; continue; }
          for (auto *named : {alias.pResourceBefore, alias.pResourceAfter}) {
            const auto identity = named ? source_cookie(named) : 0;
            if (!identity) continue;
            try {
              if (owner->aliases.capacity() == 0) owner->aliases.reserve(8);
              owner->aliases.push_back({identity, order});
              owner->aliases_sorted = false;
            } catch (...) { invalidate(*owner, recording_loss::source_state_capacity); }
          }
          continue;
        }
        if (value.Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION) {
          ++owner->native_transition_count;
          // Only subresource 0 (or all) defines the copied state; a split or
          // transition of another subresource (a stencil plane) leaves it.
          if (value.Transition.Subresource != 0 && value.Transition.Subresource != D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES) continue;
          const auto flags = value.Flags;
          // Any other flag combination is no transition this owner can follow.
          // Preserve that rejection until Reset without poisoning unrelated depth.
          if (flags != D3D12_RESOURCE_BARRIER_FLAG_NONE && flags != D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY &&
              flags != D3D12_RESOURCE_BARRIER_FLAG_END_ONLY) { block_source(*owner, value.Transition.pResource); continue; }
          // A resource can reach its first SDK nomination after all of its
          // transitions were recorded. Preserve that live object's identity
          // now, including a split that precedes the first tag/cookie. The
          // observed state remains local to this exact recording; it is never
          // a global hint.
          const auto identity = observed_source_cookie(value.Transition.pResource);
          if (!identity) { invalidate(*owner, recording_loss::source_identity_unavailable); continue; }
          auto *known = state(*owner, identity, true);
          if (!known) { invalidate(*owner, recording_loss::source_state_capacity); continue; }
          if (known->blocked) continue;
          // A legacy barrier inside an enhanced split is outside the D3D12 contract.
          if (known->pending_split && known->enhanced) { known->blocked = true; continue; }
          // So is an END_ONLY whose StateAfter is not its BEGIN_ONLY's, which
          // pending_split keeps (MS Learn, Split Barriers: the barrier that
          // ends a split has the same before and after states; the contract
          // matrix's D4c): the state is unknown.
          if (known->pending_split && flags == D3D12_RESOURCE_BARRIER_FLAG_END_ONLY &&
              value.Transition.StateAfter != known->value) { known->blocked = true; continue; }
          // BEGIN_ONLY leaves the state in transition; END_ONLY, like a
          // plain transition, completes in StateAfter.
          known->pending_split = flags == D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY;
          known->known = !known->pending_split && copy_state::supported(value.Transition.StateAfter);
          known->value = value.Transition.StateAfter;
          known->enhanced = false;
          known->order = ++owner->barrier_order;
        }
      }
    }
    void retire_recording(slot &value, std::uint64_t old) {
      if (!old) return;
      // An unsubmitted recording was discarded. Submitted resources still wait
      // for their actual queue fence; reset never proves GPU completion.
      if (value.command == old) {
        if (!value.producer_submitted) { invalidate(value, capture_failure::discarded_recording); value.finished = true; }
        value.command = 0;
        value.producer_recording.reset();
      }
      for (unsigned i = 0; i != value.consumers.size(); ++i) if (value.consumers[i] == old) {
        value.consumers[i] = 0; value.consumer_recordings[i].reset();
      }
    }
    void retire_dead_recordings(slot &value) {
      // A native-only list may never emit ReShade's destruction event. The
      // object-owned token expires without taking this mutex or retaining COM.
      // This retires CPU identities only; reclaimable still requires GPU fences.
      if (value.command && value.producer_recording &&
          value.producer_recording->cookie.load(std::memory_order_acquire) != value.command)
        retire_recording(value, value.command);
      for (unsigned i = 0; i != value.consumers.size(); ++i)
        if (value.consumers[i] && value.consumer_recordings[i] &&
            value.consumer_recordings[i]->cookie.load(std::memory_order_acquire) != value.consumers[i])
          retire_recording(value, value.consumers[i]);
    }
    // reported: ReShade's lifecycle started the new recording.
    void reset(std::uint64_t native, std::uint64_t old, HRESULT result, bool reported = false) {
      if (FAILED(result)) return;
      std::lock_guard lock(mutex);
      for_each_capture([&](auto &value) { retire_recording(value, old); });
      if (!native) return;
      auto owner = command_storage::acquire(reinterpret_cast<ID3D12Object *>(native), true);
      if (!owner) return;
      owner.life()->cookie.store(0, std::memory_order_release);
      const auto cookie = ++serial;
      if (!native_observer::set_recording_cookie(native, cookie)) {
        invalidate(*owner, recording_loss::source_identity_unavailable); return;
      }
      owner->restart(cookie, command_generation);
      owner->reported = reported;
      owner.life()->cookie.store(cookie, std::memory_order_release);
    }
    void close(std::uint64_t native, std::uint64_t cookie, HRESULT result) {
      std::lock_guard lock(mutex);
      if (auto value = command(native, cookie, true)) {
        value->closed = SUCCEEDED(result);
        if (FAILED(result)) invalidate(*value, recording_loss::close_failed);
      }
      if (FAILED(result)) for_each_capture([&](auto &value) {
        if (value.command == cookie) invalidate(value, capture_failure::close_failed);
      });
    }
    // The legacy state equivalent to a generic enhanced barrier layout, for
    // diagnostics only. Queue-specific and video layouts have none.
    std::optional<std::uint32_t> legacy_state(std::uint32_t layout) {
      switch (static_cast<D3D12_BARRIER_LAYOUT>(layout)) {
        case D3D12_BARRIER_LAYOUT_COMMON: return D3D12_RESOURCE_STATE_COMMON;
        case D3D12_BARRIER_LAYOUT_GENERIC_READ: return D3D12_RESOURCE_STATE_GENERIC_READ;
        case D3D12_BARRIER_LAYOUT_RENDER_TARGET: return D3D12_RESOURCE_STATE_RENDER_TARGET;
        case D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS: return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        case D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE: return D3D12_RESOURCE_STATE_DEPTH_WRITE;
        case D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ: return D3D12_RESOURCE_STATE_DEPTH_READ;
        case D3D12_BARRIER_LAYOUT_SHADER_RESOURCE:
          return D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        case D3D12_BARRIER_LAYOUT_COPY_SOURCE: return D3D12_RESOURCE_STATE_COPY_SOURCE;
        case D3D12_BARRIER_LAYOUT_COPY_DEST: return D3D12_RESOURCE_STATE_COPY_DEST;
        case D3D12_BARRIER_LAYOUT_RESOLVE_SOURCE: return D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
        case D3D12_BARRIER_LAYOUT_RESOLVE_DEST: return D3D12_RESOURCE_STATE_RESOLVE_DEST;
        default: return std::nullopt;
      }
    }
    // An enhanced texture barrier that includes subresource 0 sets that
    // texture's layout and access on this recording; other subresources (a
    // stencil plane) keep subresource 0's. The copy then leaves and restores
    // exactly that layout and access with enhanced barriers. A layout no copy
    // can use (neither copy_state::layouts::admissible nor a compute queue's
    // own) or NO_ACCESS refuses the copy until a later barrier names another,
    // and so does one outside the recording's list type
    // (copy_state::list_basis); a SyncAfter of SPLIT leaves the layout in
    // transition until the split's end barrier.
    void enhanced_textures(std::uint64_t native, std::uint64_t cookie, unsigned count,
        const native_observer::enhanced_texture *textures) {
      if (count && !enhanced_barriers_seen.load(std::memory_order_relaxed))
        enhanced_barriers_seen.store(true, std::memory_order_relaxed);
      // No capture lock, exactly as barriers().
      auto owner = command(native, cookie, true);
      if (!owner) return;
      if (owner->closed) { owner->missed_reset = true; return; }
      if (!owner->reported) return;
      for (unsigned i = 0; i != count; ++i) {
        const auto &texture = textures[i];
        if (!texture.resource || !texture.first_subresource) continue;
        const auto identity = observed_source_cookie(texture.resource);
        if (!identity) { invalidate(*owner, recording_loss::source_identity_unavailable); continue; }
        auto *known = state(*owner, identity, true);
        if (!known) { invalidate(*owner, recording_loss::source_state_capacity); continue; }
        known->enhanced_named = true;
        if (known->blocked) continue;
        // An enhanced barrier inside a legacy split is outside the D3D12 contract.
        if (known->pending_split && !known->enhanced) { known->blocked = true; continue; }
        const auto layout = texture.layout_after;
        known->enhanced = true;
        known->layout = layout; known->access = texture.access_after;
        known->value = legacy_state(copy_state::layouts::generic(layout)).value_or(UINT32_MAX);
        known->pending_split = texture.sync_after == D3D12_BARRIER_SYNC_SPLIT;
        known->known = !known->pending_split &&
          (copy_state::layouts::admissible(layout) || copy_state::layouts::compute_queue(layout)) &&
          texture.access_after != D3D12_BARRIER_ACCESS_NO_ACCESS;
        known->order = ++owner->barrier_order;
      }
    }
    // The capture's state evidence for a source from its entry on the recording.
    copy_state::observation observation_of(const source_state *observed) {
      return observed ? copy_state::observation{true, observed->known, observed->blocked, observed->value, observed->pending_split,
        observed->enhanced, observed->layout, observed->access} : copy_state::observation{};
    }
    // Diagnostic provenance (state_basis) of an admitted copy's pre-copy state: an entry on the
    // recording (the copy used or matched it), else the SDK contract or the
    // per-call declaration. It never changes which state the copy uses.
    state_basis copy_basis(const source_state *observed, const copy_state::decision &chosen) {
      if (observed) return observed->enhanced ? state_basis::observed_enhanced : state_basis::observed_legacy;
      return chosen.used_contract ? state_basis::contract : state_basis::declared;
    }
    void render_pass(std::uint64_t native, std::uint64_t cookie, bool inside) {
      std::lock_guard lock(mutex);
      if (auto owner = command(native, cookie, true)) owner->render_pass = inside;
    }
    void producer_submitted(slot &value, std::uint64_t native, std::uint64_t fence, bool signaled) {
      if (value.producer_submitted && value.queue != native) {
        invalidate(value, capture_failure::producer_queue_changed);
        value.retirement_unknown = true;
        return; // Never relabel the original producer's fence as another queue.
      }
      value.retirement_unknown |= !signaled;
      value.producer_submitted = true;
      value.queue = native;
      value.producer_fence = signaled ? fence : 0;
      if (!signaled) invalidate(value, capture_failure::producer_signal_failed);
      // Submission and evaluation completion are independent. A pending result
      // remains unavailable to acquire(), but a later success can admit it.
      else if (value.finished && !value.success) invalidate(value, capture_failure::evaluation_failed);
    }
    bool submitted_success(const slot &value, std::uint64_t native) {
      return value.id && !value.invalid && value.finished && value.success &&
        value.producer_submitted && value.queue == native && (value.nomination_only || value.producer_fence);
    }
    queue_progress producer_progress(const slot &value) {
      queue_progress out;
      if (const auto *owner = known_queue(value.queue)) {
        out.device_identity = owner->device_identity;
        if (!value.nomination_only && owner->fence.p) { out.completed = owner->fence->GetCompletedValue(); out.queried = true; }
      }
      return out;
    }
    bool producer_recording_retired(const slot &value) {
      return !value.command || (value.producer_recording &&
        value.producer_recording->cookie.load(std::memory_order_acquire) != value.command);
    }
    // Another queue reads a snapshot only where no later execution of its
    // producer list can write it: its copy-out (copied_out), else its own
    // storage once that recording retired (Reset or destruction; the rule
    // where no copy-out was made). Its producer fence must complete as well.
    bool immutable_for_other_queues(const slot &value) {
      return value.copied_out || producer_recording_retired(value);
    }
    // The storage every consumer reads: the copy-out once made (on the
    // producer queue it follows the copy in queue order, as later
    // submissions there do), else the snapshot's own.
    const std::shared_ptr<texture_reference> &read_storage(const slot &value) {
      return value.copied_out && value.published ? value.published : value.texture;
    }
    // U-C1 (docs/reshade-sbs.md, Queue order): another queue reads an
    // immutable snapshot once its producer fence completed, or before that
    // once the queue-order model shows that every call consumer submits from
    // now on is ordered after the snapshot's producer position (its copy-out
    // list, else the game submission of a recording that has since retired).
    // consumer 0 keeps the completion rule. The add-on never waits.
    bool readable_elsewhere(const slot &value, std::uint64_t consumer, const queue_progress &progress,
        sunshine_queue_order::verdict *relied = nullptr) {
      if (!progress.valid() || !value.producer_fence) return false;
      if (progress.completed >= value.producer_fence) return true;
      if (!consumer || value.retirement_unknown || !value.producer_position) return false;
      const auto verdict = native_observer::queue_ordered(consumer, value.producer_position);
      if (relied) *relied = verdict;
      return verdict.ordered;
    }
    // Requires mutex. Whether the newest acquisition's queue (reader) is a
    // live place.
    bool reader_live() {
      return reader && std::any_of(queues.begin(), queues.end(), [](const auto &entry) {
        return entry && entry->cookie == reader && !entry->retiring;
      });
    }
    // Requires mutex. The native queue of the newest acquisition (reader).
    std::uint64_t reader_queue() {
      for (const auto &entry : queues)
        if (entry && reader && entry->cookie == reader && !entry->retiring) return reinterpret_cast<std::uint64_t>(entry->queue.p);
      return 0;
    }
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
      const auto &current = evaluations[provider_index(provider)];
      const auto is_latest = [&](const slot &value) {
        return value.id && !value.preservation_only && value.metadata.provider == provider && value.metadata.epoch == current.epoch &&
          value.metadata.sequence == current.sequence && value.metadata.source_id == current.source_id;
      };
      for (const auto &value : slots) if (is_latest(value) && (!out.latest || value.id > out.latest->id)) out.latest = &value;
      out.selection = out.latest ? selection_reason::no_admissible_capture : selection_reason::no_current_nomination;
      if (out.latest && out.latest->queue != native) out.progress = producer_progress(*out.latest);
      unsigned active_views = 0;
      for (const auto &value : current.views)
        if (value.epoch == current.epoch && value.sequence &&
            sunshine_freshness::fresh_lt(value.tick, now, sunshine_freshness::source_age_ms)) ++active_views;
      if (active_views != 1) {
        out.result = active_views ? status::ambiguous : status::stale;
        out.selection = active_views ? selection_reason::ambiguous_views : selection_reason::inactive_views;
        return out;
      }
      const auto &pending = current.pending_nomination;
      if (pending.sequence && pending.epoch == current.epoch && pending.sequence == current.sequence &&
          pending.source_id == current.source_id && sunshine_freshness::fresh_lt(pending.tick, now, sunshine_freshness::source_age_ms) &&
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
        if (owner.provider_established && owner.provider == provider && owner.last_epoch == value.metadata.epoch &&
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
      if (!sunshine_freshness::fresh_lt(out.value->metadata.tick, now, sunshine_freshness::source_age_ms)) {
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
      if (source || chosen.pending_nomination || policy.require_frame_generation)
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
            !within_source_age(metadata.tick, now)) continue;
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
    selection_reason completed_fallback_reason(const queue_state &owner, std::uint64_t present,
        const slot &latest, const slot &completed) {
      if (completed.id <= owner.admission_after) return selection_reason::completed_before_gap;
      if (completed.metadata.sequence >= latest.metadata.sequence) return selection_reason::completed_not_older;
      if (!same_depth_storage(completed.metadata, latest.metadata)) return selection_reason::format_changed;
      // The consumed watermark belongs to the owner's provider; sequences of
      // another provider's namespace are not comparable with it.
      if (owner.provider == completed.metadata.provider && owner.last_epoch == completed.metadata.epoch &&
          completed.metadata.sequence <= owner.last_sequence)
        return selection_reason::completed_already_consumed;
      if (owner.present == present && owner.present_capture && owner.present_capture != completed.id)
        return selection_reason::presentation_already_selected;
      return selection_reason::completed_snapshot;
    }
    capture_pick select_capture(const queue_state &owner, std::uint64_t native,
        std::uint64_t present, provider_kind provider, std::uint64_t now) {
      auto chosen = select_latest_capture(owner, native, present, provider, now);
      const auto *latest = chosen.latest;
      // Missing/failed/ambiguous/expired nominations still revoke depth. A live
      // source whose newest snapshot is pending may use its own newest completed
      // one; completed_snapshot matches its identity. Ownership does not gate
      // this: a pipelined source is otherwise never readable, so it could never
      // (re)establish itself after a reset, a source change or another
      // provider's expired capture.
      if (!latest || (chosen.result != status::recorded && chosen.result != status::submitted) ||
          latest->invalid || latest->nomination_invalid || latest->nomination_only ||
          (latest->finished && !latest->success) || !within_source_age(latest->metadata.tick, now)) return chosen;
      auto *completed = completed_snapshot(latest->metadata, now, native);
      if (!completed) { chosen.selection = selection_reason::no_completed_snapshot; return chosen; }
      chosen.selection = completed_fallback_reason(owner, present, *latest, *completed);
      if (chosen.selection != selection_reason::completed_snapshot) {
        if (chosen.selection == selection_reason::completed_already_consumed &&
            completed->metadata.sequence == owner.last_sequence)
          chosen.consumed_completed_capture = completed->id;
        return chosen;
      }
      const auto progress = producer_progress(*completed);
      if (submission_status(*completed, native, owner.device_identity, progress) != status::ready) {
        chosen.selection = selection_reason::completed_not_readable; return chosen;
      }
      chosen.value = completed;
      chosen.progress = progress;
      chosen.result = status::ready;
      return chosen;
    }
    capture_pick select_provider(const queue_state &owner, std::uint64_t native,
        std::uint64_t present, std::uint64_t now, selection_policy policy = {}) {
      const auto select_sl = [&] {
        auto chosen = select_capture(owner, native, present, provider_kind::streamline, now);
        const auto *source = chosen.value ? chosen.value : chosen.latest;
        if (policy.exclude_unconfirmed_fg &&
            ((source && source->metadata.frame_generation_input) ||
              (chosen.pending_nomination && chosen.pending_nomination->source_id ==
                sunshine_scene_depth::encode_source_id(sunshine_scene_depth::fg_depth_key(chosen.pending_nomination->viewport))))) {
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
        if ((fg.value && !matches(fg.value->metadata)) || (fg.latest && !matches(fg.latest->metadata)) ||
            (fg.pending_nomination && (fg.pending_nomination->epoch != policy.epoch ||
              fg.pending_nomination->viewport != policy.viewport ||
              fg.pending_nomination->source_id != fg_source))) {
          capture_pick mismatch; mismatch.selection = selection_reason::fg_scope_mismatch; return mismatch;
        }
        return fg;
      }
      // Repeated effects in one presentation keep one immutable provider tuple.
      if (owner.provider_established && owner.present == present && owner.present_capture)
        return owner.provider == provider_kind::streamline ? select_sl() : select_capture(owner, native, present, owner.provider, now);
      auto sl = select_sl();
      auto ngx = select_capture(owner, native, present, provider_kind::ngx, now);
      const auto readable = [](const capture_pick &value) { return value.value && value.result == status::ready; };
      // Expired: no active view, or a newest nomination older than the source
      // age. A pick without any slot (an attempt rejected before its capture
      // existed) is not expired while its view is active.
      const auto expired = [&](const capture_pick &value) {
        return value.result == status::stale || (value.latest && !within_source_age(value.latest->metadata.tick, now));
      };
      // A source is live while its newest nomination is within the source age.
      // Readable implies live; liveness never extends that age.
      const auto live = [&](const capture_pick &value) {
        return value.pending_nomination || (value.latest && !expired(value));
      };
      // A live source whose newest valid snapshot is still on its way. Unlike a
      // failed, unsupported or ambiguous pick, it moves no gap watermark in
      // acquire(), so its completed snapshot can become readable.
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
      // against a readable source.
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
      // above a failing live one, which would mark a gap at every Present and
      // so keep the pending source's completed snapshots unreadable, and any
      // live pick ranks above an expired one. An owner that has stopped
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
    void consumer_submitted(slot &value, std::uint64_t native, std::uint64_t fence, bool signaled) {
      if (value.consumer_queue && value.consumer_queue != native) invalidate(value, capture_failure::consumer_queue_changed);
      if (!signaled) {
        invalidate(value, capture_failure::consumer_signal_failed); value.retirement_unknown = true; return;
      }
      auto found = std::find_if(value.retirements.begin(), value.retirements.end(),
        [native](const auto &point) { return point.queue == native; });
      if (found == value.retirements.end()) found = std::find_if(value.retirements.begin(), value.retirements.end(),
        [](const auto &point) { return !point.queue; });
      if (found == value.retirements.end()) {
        invalidate(value, capture_failure::consumer_capacity); value.retirement_unknown = true; return;
      }
      found->queue = native; found->value = std::max(found->value, fence);
    }
    // The runtime resets its immediate list after submitting it, so this one
    // observed submission (now followed by the queue fence) carries the read.
    void submitted_immediate(slot &value, std::uint32_t count, const native_observer::command_identity *values) {
      for (unsigned n = 0; n != count; ++n)
        if (value.pending_immediate && values[n].native_command == value.pending_immediate) value.pending_immediate = 0;
    }
    bool uses_queue(const slot &value, std::uint64_t native) {
      return value.queue == native || value.consumer_queue == native || std::any_of(value.retirements.begin(), value.retirements.end(),
        [native](const auto &point) { return point.queue == native; });
    }
    bool needs_submission_fence(const std::array<bool, slot_limit> &producers,
        const std::array<bool, slot_limit> &consumers) {
      for (unsigned i = 0; i != slots.size(); ++i)
        if ((!slots[i].nomination_only && producers[i]) || consumers[i]) return true;
      return false;
    }
    // Requires mutex. An aliasing barrier on the recording named source after
    // its barrier order (0: anywhere on the recording). On an executed
    // recording (sort_aliases) a binary search finds the source's last mark.
    bool aliased_after(const command_state &owner, std::uint64_t source, std::uint32_t order) {
      if (owner.aliases_sorted) {
        const auto end = std::upper_bound(owner.aliases.begin(), owner.aliases.end(), source,
          [](std::uint64_t key, const resource_mark &mark) { return key < mark.source; });
        return end != owner.aliases.begin() && std::prev(end)->source == source && std::prev(end)->order > order;
      }
      for (const auto &mark : owner.aliases) if (mark.source == source && mark.order > order) return true;
      return false;
    }
    // Requires mutex, on a recording being executed (closed, so nothing
    // appends to it until its Reset): its aliasing marks sorted once, so each
    // held texture's lookup in apply_executed costs a binary search, not a
    // scan of every mark (transient-memory engines alias every frame).
    void sort_aliases(command_state &owner) {
      if (owner.aliases_sorted) return;
      std::sort(owner.aliases.begin(), owner.aliases.end(), [](const resource_mark &a, const resource_mark &b) {
        return a.source != b.source ? a.source < b.source : a.order < b.order;
      });
      owner.aliases_sorted = true;
    }
    // Requires mutex. Whether the barrier hooks saw every barrier the list
    // recorded: its method table is hooked (states_ready). A list being
    // executed is alive, so its table pointer is a plain read; the answer is
    // remembered per table until coverage changes (command_generation), and
    // at most 16 new tables a second are checked (beyond that a list counts
    // as unobserved). Inside the submission callback, where a lost hook's
    // invalidation is suppressed like any nested notification.
    struct covered_table { std::uintptr_t table{}; std::uint64_t generation{}; bool covered{}; };
    std::array<covered_table, 8> covered_tables{};
    unsigned covered_next{}, coverage_checks{};
    std::uint64_t coverage_window{};
    bool states_covered(std::uint64_t native) {
      const auto table = *reinterpret_cast<const std::uintptr_t *>(native);
      const auto generation = command_generation.load();
      for (const auto &entry : covered_tables)
        if (entry.table == table && entry.generation == generation) return entry.covered;
      const auto now = GetTickCount64();
      if (now - coverage_window >= 1000) { coverage_window = now; coverage_checks = 0; }
      if (++coverage_checks > 16) return false;
      const bool covered = native_observer::states_ready(native);
      covered_tables[covered_next++ % covered_tables.size()] = {table, generation, covered};
      return covered;
    }
    // Requires mutex. A copy recorded from its source's proven state
    // (record_impl) is checked when its recording is executed: the proof must
    // be unchanged and the queue the one it was proven for, or the copy is
    // discarded (held, then mono). A replay is invalidated as such.
    void verify_proven_copies(std::uint64_t queue, std::uint64_t cookie) {
      if (!cookie) return;
      for_each_capture([&](slot &value) {
        if (!value.id || !value.proven_version || value.command != cookie) return;
        if (value.proven_queue != queue || proven.version(value.proven_source) != value.proven_version) {
          invalidate(value, capture_failure::proven_state_changed);
          proven_counters.changed_copies.fetch_add(1, std::memory_order_relaxed);
        }
        value.proven_version = 0;
      });
    }
    // Requires mutex. The lists of one ExecuteCommandLists call on queue, in
    // submission order (proven_state.h): copies from a proven state are
    // checked before their list's own barriers apply. A list the barrier
    // hooks did not fully observe (not reported by ReShade's lifecycle, Reset
    // without it, invalidated, or of an unhooked method table) may have
    // changed any texture: none stays proven. Otherwise every held texture
    // its barriers named takes the state its last barrier left it in, on this
    // queue, unless an aliasing barrier followed (any placed or reserved
    // resource for one naming neither), a split is still open or a barrier
    // was blocked; a depth clear after its last barrier proves DEPTH_WRITE,
    // or the DEPTH_STENCIL_WRITE layout where enhanced barriers own it. An
    // enhanced barrier naming it latches its layout (proven_state.h).
    void apply_executed(std::uint64_t queue, std::uint32_t count, const native_observer::command_identity *values) {
      for (std::uint32_t n = 0; n != count; ++n) {
        verify_proven_copies(queue, values[n].recording_cookie);
        if (!proven.size()) continue;
        const auto native = values[n].native_command;
        // A list that never had a recording state, while the hooks cover its
        // method table, recorded no barrier through them (the first would
        // have given it one): DXGI's internal Present list. It changes no
        // texture's state.
        if (!values[n].recording_cookie && native && native_observer::recording_cookie_absent(native) && states_covered(native))
          continue;
        auto owner = command(native, values[n].recording_cookie, false);
        auto &c = proven_counters;
        auto *unobserved = !owner ? &c.unrecorded : !owner->reported ? &c.unreported : owner->invalid ? &c.invalidated :
          owner->missed_reset ? &c.missed_reset : !states_covered(native) ? &c.uncovered : nullptr;
        if (unobserved) {
          proven.unprove_all();
          unobserved->fetch_add(1, std::memory_order_relaxed);
          continue;
        }
        if (owner->wildcard_alias) proven.unprove_all();
        sort_aliases(*owner);
        // Each aliased texture once (the marks are sorted by source).
        for (std::size_t i = 0; i != owner->aliases.size(); ++i)
          if (!i || owner->aliases[i].source != owner->aliases[i - 1].source) proven.unprove(owner->aliases[i].source);
        for (const auto &entry : owner->states) {
          if (!proven.held(entry.source)) continue;
          // Any enhanced barrier naming it latches its layout (D6), whatever
          // state the list left it in.
          if (entry.enhanced_named) proven.set_latch(entry.source);
          if (!entry.known || entry.blocked || entry.pending_split || owner->wildcard_alias > entry.order ||
              aliased_after(*owner, entry.source, entry.order))
            proven.unprove(entry.source);
          else if (entry.enhanced) proven.prove_layout(entry.source, queue, entry.layout, entry.access);
          else proven.prove(entry.source, queue, entry.value);
        }
        // A depth clear after a texture's last barrier proves what the D3D12
        // clear contract requires, in recording order, unless aliased after
        // it: DEPTH_WRITE, or the DEPTH_STENCIL_WRITE layout where enhanced
        // barriers own the layout (that last barrier on this list, else the
        // texture's proof; proven_state.h clear_regime). A legacy proof there
        // would let a copy record legacy barriers on an enhanced layout (D6).
        for (const auto &mark : owner->clears) {
          if (!proven.held(mark.source)) continue;
          const auto entry = std::find_if(owner->states.begin(), owner->states.end(),
            [&](const source_state &value) { return value.source == mark.source; });
          if (entry != owner->states.end() && entry->order > mark.order) continue;
          using sunshine_proven_state::regime;
          const auto kind = entry == owner->states.end() ? proven.clear_regime(mark.source) :
            entry->enhanced ? regime::enhanced : regime::legacy;
          if (kind == regime::unknown || owner->wildcard_alias > mark.order || aliased_after(*owner, mark.source, mark.order))
            proven.unprove(mark.source);
          else if (kind == regime::enhanced)
            proven.prove_layout(mark.source, queue, D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE, D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE);
          else proven.prove(mark.source, queue, D3D12_RESOURCE_STATE_DEPTH_WRITE);
        }
      }
    }
    // The capture owner's safety bounds that refused work (capacity_report),
    // one bit each, set without the lock.
    enum capacity_bound : unsigned { queue_places = 1, source_table = 2, snapshot_pool = 4, auxiliary_pool = 8 };
    std::atomic<unsigned> capacity_hits{}, capacity_reported{};
    void capacity_refused(capacity_bound bound) { capacity_hits.fetch_or(bound, std::memory_order_relaxed); }
    // Requires mutex. A place for another queue: a free one, else the place of
    // the longest unused idle queue, which no capture names, which owns no
    // provider and whose signalled fence values have all completed (the
    // device's removal completes them). The idle queue registers again when
    // it next carries a copy (register_compute_producer) or presents
    // (observe_present), so no number of queues created over a session
    // exhausts the table.
    std::unique_ptr<queue_state> *free_place() {
      for (auto &entry : queues) if (!entry) return &entry;
      std::unique_ptr<queue_state> *idle = nullptr;
      for (auto &entry : queues) {
        if (entry->retiring || entry->provider_established || (idle && (*idle)->last_use <= entry->last_use)) continue;
        const auto native = reinterpret_cast<std::uint64_t>(entry->queue.p);
        bool used = false;
        for_each_capture([&](const slot &value) { used = used || (value.id && uses_queue(value, native)); });
        if (!used && (!entry->fence.p || entry->fence->GetCompletedValue() >= entry->next_fence)) idle = &entry;
      }
      return idle;
    }
    // Protected by mutex: an idle queue's place was reclaimed (free_place), so
    // a queue with no place may carry a copy (register_compute_producer).
    bool places_reclaimed{};
    // Requires mutex. A queue's place: its device and the fence the capture
    // owner signals after the queue's submissions that carry a copy. DIRECT
    // queues register as ReShade reports them (observe_queue);
    // compute_producer also admits a COMPUTE queue, which only ever produces
    // (a copy recorded on a compute list: S22, N7).
    queue_state *register_queue(ID3D12CommandQueue *value, bool compute_producer) {
      const auto type = value->GetDesc().Type;
      if (type != D3D12_COMMAND_LIST_TYPE_DIRECT && !(compute_producer && type == D3D12_COMMAND_LIST_TYPE_COMPUTE)) {
        reason = status::unsupported_queue; return nullptr;
      }
      auto *entry = free_place();
      if (!entry) {
        capacity_refused(queue_places);
        reason = status::exhausted;
        return nullptr;
      }
      if (*entry) { entry->reset(); places_reclaimed = true; }
      auto next = std::make_unique<queue_state>();
      if (FAILED(value->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void **>(next->device.put()))) ||
          FAILED(next->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), reinterpret_cast<void **>(next->fence.put())))) {
        reason = status::unsupported_queue; return nullptr;
      }
      next->device_identity = device_cookie(next->device.p, true);
      if (!next->device_identity) { reason = status::unsupported_queue; return nullptr; }
      value->AddRef(); next->queue.p = value;
      next->compute_producer = type == D3D12_COMMAND_LIST_TYPE_COMPUTE;
      next->cookie = ++serial;
      next->last_use = ++place_clock;
      if (FAILED(value->SetPrivateData(queue_guid, sizeof(next->cookie), &next->cookie))) { reason = status::unsupported_queue; return nullptr; }
      *entry = std::move(next); return entry->get();
    }
    // Requires mutex. A COMPUTE queue that ReShade reported (or not) takes a
    // place at its first submission of a copy awaiting its producer: a copy
    // recorded on a compute list (record_impl). Without one, such a copy is
    // never submitted, and its recording's Reset discards it. So does any
    // queue whose idle place was reclaimed. While a compute queue found no
    // place (compute_unplaced), record_impl refuses compute-list copies until
    // one is free again (collect_retired, free_place).
    bool compute_copies_recorded{}; // Protected by mutex: a copy was ever recorded on a COMPUTE list.
    bool compute_unplaced{}; // Protected by mutex.
    bool compute_producer_placed() {
      return std::any_of(queues.begin(), queues.end(),
        [](const auto &value) { return value && !value->retiring && value->compute_producer; });
    }
    queue_state *register_compute_producer(std::uint64_t native, std::uint32_t count,
        const native_observer::command_identity *values) {
      if (!compute_copies_recorded && !places_reclaimed) return nullptr;
      const auto awaits = [&](const auto &captures) {
        for (std::uint32_t n = 0; n != count; ++n) {
          const auto cookie = values[n].recording_cookie;
          if (!cookie) continue;
          for (const auto &value : captures)
            if (value.id && !value.producer_submitted && value.command == cookie) return true;
        }
        return false;
      };
      if (!awaits(slots) && !(diagnostic_slots_active && awaits(diagnostic_slots))) return nullptr;
      com_ptr<ID3D12CommandQueue> checked;
      if (!query_native(native, IID_ID3D12CommandQueue, checked)) return nullptr;
      auto *owner = register_queue(checked.p, true);
      if (!owner && reason == status::exhausted) compute_unplaced = true;
      return owner;
    }
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
    // Requires mutex. Depth is read on a queue other than owner: the newest
    // acquisition (reader) was another live queue's. Not whether owner ever
    // acquired: a render queue that presented with FG off reads elsewhere
    // once FG presents on DLSS-G's queue, and no longer once FG is off again.
    bool read_elsewhere(const queue_state &owner) {
      return reader && reader != owner.cookie && std::any_of(queues.begin(), queues.end(), [&](const auto &entry) {
        return entry && entry->cookie == reader && !entry->retiring;
      });
    }
    // U-C1b: FG options turned on within the source age, so Present may move
    // to DLSS-G's queue at FG's first Present, after the frame's copies on
    // the queue that reads depth now were submitted (until 10-27 those kept
    // the Reset rule, and that first FG Present was flat). Bounded: a game
    // whose Present stays on its queue with FG on pays one copy-out per copy
    // for that age after each FG On.
    bool reader_may_move(std::uint64_t now) {
      const auto on = frame_generation_on_tick.load(std::memory_order_relaxed);
      return on && sunshine_freshness::fresh_lt(on, now, sunshine_freshness::source_age_ms);
    }
    // Requires mutex. Copy-out on the producer queue (docs/reshade-sbs.md):
    // in the submission hook, right after the game's ExecuteCommandLists and
    // before the producer fence value (fence) is signalled. A first
    // submission of a copy that another queue reads (read_elsewhere) stamps
    // its source's foreign_tick, so that its copies recorded within the
    // source age take copy-out storage (record_impl,
    // read_elsewhere_recently); each such copy is copied into that storage
    // with one list of the add-on's own on this queue, at most one per game
    // submission. Returns the slots copied out; the caller sets copied_out
    // once the fence after them was signalled. A copy not copied out keeps
    // the Reset rule (immutable_for_other_queues).
    bool reserved_by_other(const slot &entry, std::uint64_t own);
    // Requires mutex. U-C1b: a copy submitted on a queue other than the
    // reader without copy-out storage (recorded before its source was read
    // elsewhere, as when FG On moves Present to DLSS-G's queue) takes an idle
    // slot's cached storage of its snapshot's shape, format, view and device
    // (rest COMMON on a COMPUTE queue), so the game's submission thread never
    // allocates under the capture lock. The idle slot allocates again at its
    // next record, without the lock. False when no idle slot holds such
    // storage: the copy keeps the Reset rule, as before.
    bool take_idle_storage(slot &value, bool compute) {
      const auto &own = *value.texture;
      const auto shape = own.resource->GetDesc();
      const auto fits = [&](const std::shared_ptr<texture_reference> &storage) {
        if (!storage || storage.use_count() != 1 || !storage->resource.p || storage->device_identity != own.device_identity ||
            storage->view_format != own.view_format || (compute && storage->rest != D3D12_RESOURCE_STATE_COMMON))
          return false;
        const auto desc = storage->resource->GetDesc();
        return desc.Width == shape.Width && desc.Height == shape.Height && desc.Format == shape.Format;
      };
      for (auto &entry : slots) {
        if (entry.id || reserved_by_other(entry, 0)) continue;
        for (auto *storage : {&entry.published, &entry.texture})
          if (fits(*storage)) {
            value.published = std::move(*storage);
            return true;
          }
      }
      return false;
    }
    std::array<bool, slot_limit> copy_out(queue_state &owner, const std::array<bool, slot_limit> &producers,
        std::uint64_t fence) {
      std::array<bool, slot_limit> out{};
      if (!fence) return out;
      const auto now = GetTickCount64();
      // The reader's own queue copies out too while FG may move Present off
      // it (reader_may_move); only a copy read elsewhere stamps its source.
      const bool elsewhere = read_elsewhere(owner);
      if (!elsewhere && !(reader == owner.cookie && !owner.compute_producer && reader_may_move(now))) return out;
      bool any = false;
      for (unsigned i = 0; i != slots.size(); ++i) {
        auto &value = slots[i];
        if (!producers[i] || value.nomination_only || value.preservation_only || value.producer_submitted) continue;
        if (elsewhere && value.metadata.source) value.metadata.source->foreign_tick.store(now, std::memory_order_relaxed);
        if (value.invalid || !value.texture || value.copied_out) continue;
        if (!value.published) {
          const bool taken = take_idle_storage(value, owner.compute_producer);
          (taken ? copy_out_counters.taken : copy_out_counters.unstored).fetch_add(1, std::memory_order_relaxed);
          if (!taken) continue;
        }
        // A COMPUTE queue names only COMMON of these (record_impl's rest).
        if (owner.compute_producer && (value.texture->rest != D3D12_RESOURCE_STATE_COMMON ||
            value.published->rest != D3D12_RESOURCE_STATE_COMMON)) continue;
        out[i] = any = true;
      }
      if (!any) return out;
      const auto refuse = [&](std::atomic<std::uint64_t> &counter) {
        counter.fetch_add(1, std::memory_order_relaxed);
        out.fill(false);
        return out;
      };
      // A list whose last execution completed (its fence value passed).
      const auto completed = owner.fence->GetCompletedValue();
      copy_out_list *entry = nullptr;
      for (auto &candidate : owner.copy_out)
        if (!candidate.list.p || (completed != UINT64_MAX && completed >= candidate.fence)) { entry = &candidate; break; }
      if (!entry) return refuse(copy_out_counters.ring_busy);
      const auto type = owner.compute_producer ? D3D12_COMMAND_LIST_TYPE_COMPUTE : D3D12_COMMAND_LIST_TYPE_DIRECT;
      const bool opened = entry->list.p ?
        SUCCEEDED(entry->allocator->Reset()) && SUCCEEDED(entry->list->Reset(entry->allocator.p, nullptr)) :
        SUCCEEDED(owner.device->CreateCommandAllocator(type, __uuidof(ID3D12CommandAllocator),
          reinterpret_cast<void **>(entry->allocator.put()))) &&
        SUCCEEDED(owner.device->CreateCommandList(0, type, entry->allocator.p, nullptr, __uuidof(ID3D12GraphicsCommandList),
          reinterpret_cast<void **>(entry->list.put())));
      auto *list = entry->list.p;
      if (opened) for (unsigned i = 0; i != slots.size(); ++i) {
        if (!out[i]) continue;
        const auto &value = slots[i];
        D3D12_RESOURCE_BARRIER transitions[2]{};
        transitions[0].Type = transitions[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        transitions[0].Transition = {value.texture->resource.p, 0, value.texture->rest, D3D12_RESOURCE_STATE_COPY_SOURCE};
        transitions[1].Transition = {value.published->resource.p, 0, value.published->rest, D3D12_RESOURCE_STATE_COPY_DEST};
        list->ResourceBarrier(2, transitions);
        list->CopyResource(value.published->resource.p, value.texture->resource.p);
        for (auto &transition : transitions) std::swap(transition.Transition.StateBefore, transition.Transition.StateAfter);
        list->ResourceBarrier(2, transitions);
      }
      if (!opened || FAILED(list->Close())) {
        entry->list.reset(); entry->allocator.reset(); // Not executed since its last fence completed.
        return refuse(copy_out_counters.failed);
      }
      ID3D12CommandList *lists[]{list};
      owner.queue->ExecuteCommandLists(1, lists); // The hooks are suppressed in this callback.
      entry->fence = fence;
      return out;
    }
    void submitted(std::uint64_t native, std::uint32_t count, const native_observer::command_identity *values) {
      // This game submission's queue-order position, read before any list of
      // the add-on's own runs on this thread (copy_out takes the next one).
      const auto game_at = native_observer::last_execute_position();
      std::lock_guard lock(mutex);
      apply_executed(native, count, values);
      auto *owner = queue(native);
      if (!owner) owner = register_compute_producer(native, count, values);
      if (!owner) return;
      native = reinterpret_cast<std::uint64_t>(owner->queue.p);
      std::array<bool, slot_limit> producers{}, consumers{};
      const bool depth_used = identify_submissions(slots, count, values, producers, consumers);
      std::array<bool, diagnostic_slot_limit> auxiliary_producers, auxiliary_consumers;
      bool auxiliary_used = false;
      if (diagnostic_slots_active) {
        auxiliary_producers.fill(false); auxiliary_consumers.fill(false);
        auxiliary_used = identify_submissions(diagnostic_slots, count, values, auxiliary_producers, auxiliary_consumers);
      }
      if (!depth_used && !auxiliary_used) return;
      // The queue of each list that carried a depth copy, which its next
      // copy's copy-out storage is predicted from (U-C1b, record_impl): one
      // private-data read per such list.
      if (depth_used) for (std::uint32_t n = 0; n != count; ++n) {
        const auto cookie = values[n].recording_cookie;
        bool carried = false;
        for (unsigned i = 0; i != slots.size() && !carried; ++i) carried = producers[i] && cookie && slots[i].command == cookie;
        if (!carried) continue;
        if (auto list = command_storage::acquire(reinterpret_cast<ID3D12Object *>(values[n].native_command), false))
          list->executed_queue = owner->cookie;
        if (!owner->compute_producer) last_direct_copy_queue = owner->cookie;
      }
      const bool needs_fence = auxiliary_used || needs_submission_fence(producers, consumers);
      const auto fence = needs_fence ? ++owner->next_fence : 0;
      const auto copied = depth_used ? copy_out(*owner, producers, fence) : std::array<bool, slot_limit>{};
      const auto copy_at = std::find(copied.begin(), copied.end(), true) != copied.end() ?
        native_observer::last_execute_position() : native_observer::position{};
      // This callback runs AFTER the actual native ExecuteCommandLists. Signal
      // therefore follows the copied/consumed pixels (and copy-outs) on this queue.
      const bool signaled = !needs_fence || SUCCEEDED(owner->queue->Signal(owner->fence.p, fence));
      if (!signaled)
        for (auto &entry : owner->copy_out) if (entry.fence == fence) entry.fence = UINT64_MAX; // Never reused.
      for (unsigned i = 0; i != slots.size(); ++i) {
        auto &value = slots[i];
        if (producers[i]) {
          producer_submitted(value, native, value.nomination_only ? 0 : fence, value.nomination_only || signaled);
          // The position another queue's read must be ordered after (U-C1):
          // the copy-out list, else this submission (read only once its
          // recording retired). A replay moves it, as it moves the fence.
          value.producer_position = copied[i] && signaled ? copy_at : game_at;
          if (copied[i] && signaled) {
            value.copied_out = true;
            copy_out_counters.made.fetch_add(1, std::memory_order_relaxed);
          }
          // An executed copy-out without its fence is never released unused.
          if (copied[i]) value.predicted_storage = false;
          accept_compute_copy(value);
        }
        if (consumers[i]) {
          consumer_submitted(value, native, fence, signaled);
          // Keep the recording reference until Reset/destroy. A legal replay of
          // an unchanged command list can consume this texture again.
          submitted_immediate(value, count, values);
        }
      }
      if (auxiliary_used) for (unsigned i = 0; i != diagnostic_slots.size(); ++i) {
        if (auxiliary_producers[i]) producer_submitted(diagnostic_slots[i], native, fence, signaled);
        if (auxiliary_consumers[i]) {
          consumer_submitted(diagnostic_slots[i], native, fence, signaled);
          submitted_immediate(diagnostic_slots[i], count, values);
        }
      }
    }
    bool current_source_nomination(const slot &value) {
      if (!value.id || !value.source_nominated || value.preservation_only || value.invalid ||
          value.nomination_invalid || (value.finished && !value.success) || !valid_provider(value.metadata.provider))
        return false;
      const auto &head = evaluations[provider_index(value.metadata.provider)];
      return value.metadata.epoch == head.epoch && value.metadata.sequence == head.sequence &&
        value.metadata.source_id == head.source_id;
    }
    enum class source_retention { none, current_nomination, completed_fallback };
    source_retention logical_source_retention(slot &value) {
      if (value.nomination_only) {
        // Metadata-only nominations carry source authority but no GPU storage.
        const auto &head = evaluations[provider_index(value.metadata.provider)];
        const bool current = !value.invalid && (!value.finished || value.success) &&
          value.metadata.epoch == head.epoch && value.metadata.sequence == head.sequence &&
          value.metadata.source_id == head.source_id;
        return current ? source_retention::current_nomination : source_retention::none;
      }
      if (!value.source_nominated || value.preservation_only || value.invalid || value.nomination_invalid ||
          !valid_provider(value.metadata.provider)) return source_retention::none;
      // GPU retirement does not retire the provider's nomination authority.
      // A different provider (or Generic preservation) shares this pool and
      // must not erase the current head while its namespace still names it.
      if (current_source_nomination(value)) return source_retention::current_nomination;
      const slot *latest = nullptr;
      for (const auto &candidate : slots)
        if (candidate.metadata.provider == value.metadata.provider && current_source_nomination(candidate) &&
            (!latest || candidate.id > latest->id)) latest = &candidate;
      // Preserve exactly the existing eligible completed fallback too. Its
      // age, reset/revision, source and GPU-completion requirements remain in
      // completed_snapshot; retaining storage grants no extra admission.
      return latest && completed_snapshot(latest->metadata, GetTickCount64()) == &value ?
        source_retention::completed_fallback : source_retention::none;
    }
    bool snapshot_storage_retired(const slot &value) {
      if (!value.id) return true;
      if (value.diagnostic_only && !value.diagnostic_released) return false;
      // No copy allocation or submitted GPU obligation belongs to metadata.
      if (value.nomination_only) return true;
      if (value.retirement_unknown) return false;
      if (value.command) return false; // A closed producer may legally replay.
      if (value.texture.use_count() > 1 || value.published.use_count() > 1) return false;
      if (std::any_of(value.consumers.begin(), value.consumers.end(), [](auto v) { return v != 0; })) return false;
      if (value.pending_immediate) return false;
      if (!value.producer_submitted) return value.finished && value.command == 0;
      return gpu_retired(value, [](const fence_point &point) {
        const auto *owner = known_queue(point.queue);
        if (!owner || !owner->fence.p) return false;
        const auto complete = owner->fence->GetCompletedValue();
        return complete != UINT64_MAX && complete >= point.value;
      });
    }
    bool reclaimable(slot &value) {
      retire_dead_recordings(value);
      if (!value.id) return true;
      // Logical source lifetime and GPU storage lifetime are independent. A
      // retired allocation can still carry authority; withdrawing authority
      // never discharges an unfinished recording, CPU lease or queue fence.
      return logical_source_retention(value) == source_retention::none && snapshot_storage_retired(value);
    }
    void collect_diagnostics() {
      if (!diagnostic_slots_active) return;
      bool retained = false;
      const auto now = GetTickCount64();
      for (auto &value : diagnostic_slots) {
        if (value.id && reclaimable(value)) {
          // Retire the capture identity/leases, not its reusable allocation.
          // Live FG alpha uses this owner every real frame; destroying storage
          // here would turn the bounded ring into a per-frame GPU allocator.
          auto storage = value.diagnostic_reusable ? std::move(value.texture) : nullptr;
          value = {};
          value.texture = std::move(storage);
          value.idle_tick = now;
        }
        retained |= value.id != 0;
      }
      diagnostic_slots_active = retained;
    }
    constexpr std::uint64_t reservation_timeout_ms = 1000;
    // Reservation tokens never consume capture IDs (serial). Protected by mutex.
    std::uint64_t reservation_serial{};
    // Requires the lock. A live reservation of another record attempt.
    bool reserved_by_other(const slot &entry, std::uint64_t own) {
      return entry.reservation && entry.reservation != own &&
        GetTickCount64() - entry.reservation_tick < reservation_timeout_ms;
    }
    // Requires the lock.
    void clear_reservation(std::uint64_t token) {
      if (!token) return;
      const auto clear = [token](slot &entry) {
        if (entry.reservation == token) { entry.reservation = entry.reservation_tick = entry.reserved_bytes = 0; entry.reserved_live = false; }
      };
      for (auto &entry : slots) clear(entry);
      for (auto &entry : diagnostic_slots) clear(entry);
    }
    // Live snapshots (record_local_texture, reused every frame) and Dump 3D
    // copies keep separate budgets, so an armed dump never evicts or crowds
    // live storage. Dump copies keep the fixed limit. Live storage is sized
    // from the request: room for this many snapshots of its size (a few UI
    // kinds, each with a frame in flight), never less than the fixed limit.
    constexpr std::uint64_t live_snapshot_reserve = 8;
    // Idle cached storage is always live: dump copies are never cached.
    bool live_storage(const slot &entry) { return entry.id ? entry.diagnostic_reusable : bool(entry.texture); }
    // own: the caller's reservation, whose reserved bytes are its own. Evicted
    // idle storage moves to discard (released by the caller after the lock).
    bool reserve_diagnostic_bytes(std::array<slot, diagnostic_slot_limit> &pool,
        const slot *replacement, std::uint64_t bytes, bool live, std::uint64_t own = 0,
        std::vector<std::shared_ptr<texture_reference>> *discard = nullptr) {
      const auto limit = live ? std::max(diagnostic_byte_limit, live_snapshot_reserve * bytes) : diagnostic_byte_limit;
      if (!bytes || bytes > limit) return false;
      std::uint64_t allocated = 0;
      for (const auto &entry : pool) {
        if (&entry != replacement && entry.texture && live_storage(entry) == live) allocated += entry.texture->allocation_bytes;
        // Storage another attempt is allocating without the lock right now.
        if (reserved_by_other(entry, own) && entry.reserved_live == live) allocated += entry.reserved_bytes;
      }
      // A format/size change may need different storage. Evict only idle cache
      // entries; live tickets and any unfinished GPU obligations retain theirs.
      for (auto &entry : pool) {
        if (allocated <= limit - bytes) return true;
        if (live && &entry != replacement && !entry.id && entry.texture) {
          allocated -= entry.texture->allocation_bytes;
          if (discard) discard->push_back(std::move(entry.texture));
          entry.texture.reset();
        }
      }
      return allocated <= limit - bytes;
    }
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
    // Never called with the capture lock: the driver may take milliseconds.
    // The device is the recording's retained source's, alive while the game
    // records with it; release_device releases this storage before it goes.
    record_stage allocate_storage(const storage_plan &plan, std::shared_ptr<texture_reference> &out) {
      out.reset();
      try {
        auto texture = std::make_shared<texture_reference>();
        texture->device = plan.device;
        texture->device_identity = plan.device_identity;
        texture->allocation_bytes = plan.allocation_bytes;
        texture->rest = plan.state;
        D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        native_observer::suppression_scope suppress;
        if (FAILED(plan.device->CreateCommittedResource(&heap, plan.shared ? D3D12_HEAP_FLAG_SHARED : D3D12_HEAP_FLAG_NONE,
            &plan.desc, plan.state, nullptr, __uuidof(ID3D12Resource), reinterpret_cast<void **>(texture->resource.put()))))
          return record_stage::texture_allocation;
        texture->identity = retain_source_cookie(texture->resource.p);
        if (!texture->identity) return record_stage::texture_allocation;
        if (plan.shared &&
            FAILED(plan.device->CreateSharedHandle(texture->resource.p, nullptr, GENERIC_ALL, nullptr, &texture->shared_handle)))
          return record_stage::texture_allocation;
        if (plan.descriptors) {
          D3D12_DESCRIPTOR_HEAP_DESC descriptors{};
          descriptors.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; descriptors.NumDescriptors = 1;
          if (FAILED(plan.device->CreateDescriptorHeap(&descriptors, __uuidof(ID3D12DescriptorHeap),
              reinterpret_cast<void **>(texture->views.put())))) return record_stage::descriptor_allocation;
          if (plan.view != DXGI_FORMAT_UNKNOWN) {
            const auto handle = texture->views->GetCPUDescriptorHandleForHeapStart();
            D3D12_SHADER_RESOURCE_VIEW_DESC view{};
            view.Format = plan.view; view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; view.Texture2D.MipLevels = 1;
            plan.device->CreateShaderResourceView(texture->resource.p, &view, handle);
            texture->shader_resource = handle.ptr;
            texture->view_format = plan.view;
          }
        }
        out = std::move(texture);
        return record_stage::recorded;
      } catch (...) {
        return record_stage::texture_allocation;
      }
    }
    // Requires mutex. New storage joins the registry (storage).
    bool register_storage(const std::shared_ptr<texture_reference> &value) {
      try {
        if (storage.size() == storage.capacity())
          storage.erase(std::remove_if(storage.begin(), storage.end(), [](const auto &entry) { return entry.expired(); }),
            storage.end());
        storage.push_back(value);
        return true;
      } catch (...) {
        return false;
      }
    }
    // A metadata-only nomination: source authority without GPU storage.
    std::uint64_t record_nomination(const input &value, std::uint64_t normalized_source,
        std::uint64_t cookie, const recording_ref &recording) {
      // Keep the GPU-capable pool available while trying alternate tags in one
      // evaluation. A metadata-only high-resolution nomination must not occupy
      // the last slot that could capture the ordinary depth tag.
      const auto pixel_end = slots.begin() + pixel_slot_limit;
      auto found = std::find_if(pixel_end, slots.end(), reclaimable);
      if (found == slots.end()) {
        found = std::find_if(slots.begin(), pixel_end, reclaimable);
        if (found == pixel_end) return 0;
      }
      *found = {};
      found->nomination_only = true;
      found->metadata = value; found->metadata.resource.native = normalized_source;
      found->id = ++serial; found->command = cookie; found->producer_recording = recording;
      return found->id;
    }
    bool reusable_preserved(const slot &entry, const input &value, std::uint64_t recording) {
      return entry.id && entry.preservation_only && !entry.invalid && !entry.producer_submitted &&
        !entry.acquired && entry.command == recording && entry.metadata.source && value.source &&
        entry.metadata.source->cookie == value.source->cookie &&
        std::none_of(entry.consumers.begin(), entry.consumers.end(), [](auto id) { return id != 0; });
    }
    void describe_packet(const slot &value, std::uint64_t consumer, packet &out) {
      out = {};
      // Every consumer reads the copy-out once made (read_storage).
      const auto &storage = read_storage(value);
      const auto desc = value.nomination_only ? value.metadata.source->desc : storage->resource->GetDesc();
      out.metadata = value.metadata; out.ownership = storage;
      out.resource_id = value.metadata.source ? value.metadata.source->cookie : 0;
      if (value.nomination_only) out.device = reinterpret_cast<std::uint64_t>(value.metadata.source->device);
      else {
        out.texture = reinterpret_cast<std::uint64_t>(storage->resource.p);
        out.shader_resource = storage->shader_resource;
        out.device = reinterpret_cast<std::uint64_t>(storage->device);
      }
      out.queue = consumer; out.producer_queue = value.queue; out.capture_id = value.id; out.producer_fence = value.producer_fence;
      out.width = static_cast<std::uint32_t>(desc.Width); out.height = desc.Height;
      out.area = sunshine_api_contract::normalized_extent(value.metadata.resource.area, out.width, out.height);
      out.format = typeless(desc.Format);
      // The snapshot's view was resolved at capture (depth_view); a
      // metadata-only nomination resolves its source's the same way.
      out.srv_format = value.nomination_only ? depth_view(desc.Format, value.metadata) : storage->view_format;
      out.pixel_ready = !value.nomination_only;
    }
    void collect_retired() {
      for (auto &owner : queues) {
        if (!owner || !owner->retiring) continue;
        const auto native = reinterpret_cast<std::uint64_t>(owner->queue.p);
        const bool removed = FAILED(owner->device->GetDeviceRemovedReason());
        bool retained = false;
        for_each_capture([&](auto &value) {
          if (value.id && uses_queue(value, native)) {
            if (removed || reclaimable(value)) value = {};
            else retained = true;
          }
        });
        if (!retained) {
          owner.reset();
          compute_unplaced = false;
        }
      }
      collect_diagnostics();
    }
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
  void shutdown() {
    requested = false;
    { std::lock_guard lock(mutex);
      for (auto &owner : queues) if (owner) owner->provider_established = false; }
    native_observer::shutdown();
    invalidate_all();
    // Pinned hooks and bounded in-flight allocations remain alive: unloading or
    // releasing uncompleted GPU work would be unsafe. No shutdown wait is added.
    reason = status::inactive;
  }
  bool active() {
    if (!requested.load()) return false;
    std::lock_guard lock(mutex);
    return std::any_of(queues.begin(), queues.end(), [](const auto &value) { return value && !value->retiring; });
  }
  void observe_command(std::uint64_t native) {
    if (!requested.load() || !native) return;
    com_ptr<ID3D12GraphicsCommandList> checked;
    if (!query_native(native, IID_ID3D12GraphicsCommandList, checked)) return;
    native = reinterpret_cast<std::uint64_t>(checked.p);
    auto cookie = native_observer::get_recording_cookie(native);
    // Nomination uses the same strict first-recording association as native
    // method entry. Failed/malformed private data must never be revived here.
    if (!cookie) associate_recording(native, &cookie);
    {
      std::lock_guard lock(mutex);
      // Attach before hook discovery can invalidate the current recording.
      // Late payload allocation must not erase a loss already observed here.
      command(native, cookie, true);
    }
    native_observer::observe_command(native);
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
    bool runtime_list(std::uint64_t native) {
      if (!native || !runtime_lists_known.load(std::memory_order_acquire)) return false;
      AcquireSRWLockShared(&runtime_lock);
      const bool found = std::find(runtime_lists.begin(), runtime_lists.end(), native) != runtime_lists.end();
      ReleaseSRWLockShared(&runtime_lock);
      return found;
    }
    // ReShade reports only game lists, so a reported list at a runtime list's
    // address is a game list that reused it.
    void forget_runtime_list(std::uint64_t native) {
      if (!runtime_list(native)) return;
      AcquireSRWLockExclusive(&runtime_lock);
      for (auto &value : runtime_lists) if (value == native) value = 0;
      ReleaseSRWLockExclusive(&runtime_lock);
    }
    // ReShade's lifecycle is the only source of a list's recording. created: a
    // recording opens (a new list's first, or a registered runtime list's
    // current one); reset: the previous one ends and the next opens. native is
    // the list object itself, whose private data holds its recording.
    void lifecycle_recording(std::uint64_t native, list_event event) {
      const auto cookie = native_observer::get_recording_cookie(native);
      switch (event) {
        case list_event::created:
          if (cookie) { reset(native, cookie, S_OK, true); break; }
          {
            std::uint64_t first{};
            associate_recording(native, &first);
            std::lock_guard lock(mutex);
            if (auto owner = command(native, first, true)) owner->reported = true;
          }
          break;
        case list_event::reset: reset(native, cookie, S_OK, true); break;
        case list_event::closed: close(native, cookie, S_OK); break;
        case list_event::pass_begin: render_pass(native, cookie, true); break;
        case list_event::pass_end: render_pass(native, cookie, false); break;
        default: break;
      }
    }
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
    void raise_execution(queue_execution &entry, std::uint64_t newest) {
      std::atomic_ref<std::uint64_t> value(entry.newest);
      auto seen = value.load(std::memory_order_relaxed);
      while (seen < newest && !value.compare_exchange_weak(seen, newest, std::memory_order_relaxed)) {}
    }
    queue_execution *find_queue_execution(std::uint64_t queue) {
      for (std::size_t i = 0; i != queue_execution_count; ++i)
        if (queue_executions[i].queue == queue) return &queue_executions[i];
      return nullptr;
    }
    void note_queue_execution(std::uint64_t queue, std::uint64_t newest) {
      if (!queue) return;
      AcquireSRWLockShared(&queue_execution_lock);
      auto *known = find_queue_execution(queue);
      if (known) raise_execution(*known, newest);
      ReleaseSRWLockShared(&queue_execution_lock);
      if (known) return;
      AcquireSRWLockExclusive(&queue_execution_lock);
      auto *entry = find_queue_execution(queue);
      if (!entry) {
        if (queue_execution_count != queue_executions.size()) entry = &queue_executions[queue_execution_count++];
        else {
          entry = &queue_executions.front();
          for (auto &value : queue_executions) if (value.newest < entry->newest) entry = &value;
        }
        *entry = {queue, 0};
      }
      entry->newest = std::max(entry->newest, newest);
      ReleaseSRWLockExclusive(&queue_execution_lock);
    }
    void forget_queue_execution(std::uint64_t queue) {
      if (!queue) return;
      AcquireSRWLockExclusive(&queue_execution_lock);
      if (auto *entry = find_queue_execution(queue)) {
        *entry = queue_executions[--queue_execution_count];
        queue_executions[queue_execution_count] = {};
      }
      ReleaseSRWLockExclusive(&queue_execution_lock);
    }
    void observed_submission(std::uint64_t queue, std::uint32_t count, const native_observer::command_identity *values) {
      submitted(queue, count, values);
      // ReShade resets its own list right after submitting it: the submitted
      // recording ends and the next one begins. Any other list with a
      // recording is the game's (game_executions).
      bool game = false;
      for (std::uint32_t i = 0; i != count; ++i)
        if (runtime_list(values[i].native_command))
          try { lifecycle_recording(values[i].native_command, list_event::reset); } catch (...) {}
        else if (values[i].recording_cookie) game = true;
      if (game) note_queue_execution(queue, game_execution_count.fetch_add(1, std::memory_order_acq_rel) + 1);
    }
  }
  void observe_list_event(std::uint64_t native, list_event event) {
    if (!native || event >= list_event::count) return;
    switch (event) {
      case list_event::destroyed: command_destroyed(native); return;
      case list_event::created: forget_runtime_list(native); break;
      default: break;
    }
    if (!requested.load()) return;
    try { lifecycle_recording(native, event); } catch (...) {}
  }
  void observe_runtime_list(std::uint64_t native, std::uint64_t queue) {
    if (!native || !requested.load() || runtime_list(native)) return;
    // Its recordings advance only through submissions observed on its queue.
    if (!native_observer::queue_ready(queue)) return;
    com_ptr<ID3D12GraphicsCommandList> checked;
    if (!query_native(native, IID_ID3D12GraphicsCommandList, checked) || reinterpret_cast<std::uint64_t>(checked.p) != native) return;
    bool added = false;
    AcquireSRWLockExclusive(&runtime_lock);
    if (std::find(runtime_lists.begin(), runtime_lists.end(), native) == runtime_lists.end())
      for (auto &value : runtime_lists) if (!value) { value = native; added = true; break; }
    if (added) runtime_lists_known.store(true, std::memory_order_release);
    ReleaseSRWLockExclusive(&runtime_lock);
    // Registered while ReShade records it: its current recording is open.
    if (added) try { lifecycle_recording(native, list_event::created); } catch (...) {}
  }
  void observe_present(std::uint64_t command_native, std::uint64_t queue_native, std::uint64_t immediate) {
    if (!requested.load()) return;
    // The registered immediate list was checked for this exact interface at
    // registration; any other list (or an unregistered one) is queried.
    std::uint64_t list{}, cookie{};
    if (command_native && command_native == immediate && runtime_list(command_native)) {
      list = command_native;
      cookie = native_observer::get_recording_cookie(list);
    }
    if (!list && command_native) {
      com_ptr<ID3D12GraphicsCommandList> checked;
      if (query_native(command_native, IID_ID3D12GraphicsCommandList, checked)) {
        list = reinterpret_cast<std::uint64_t>(checked.p);
        cookie = native_observer::get_recording_cookie(list);
      }
    }
    // The same strict first-recording association as observe_command.
    if (list && !cookie) associate_recording(list, &cookie);
    bool known_queue_present = false;
    {
      std::lock_guard lock(mutex);
      if (list) command(list, cookie, true);
      if (queue_native && known_queue(queue_native)) {
        known_queue_present = true;
        collect_retired();
      }
    }
    if (list) native_observer::observe_command(list);
    if (queue_native) {
      if (known_queue_present) native_observer::observe_queue(queue_native);
      else observe_queue(queue_native);
    }
    if (immediate) observe_runtime_list(immediate, queue_native);
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
  void command_destroyed(std::uint64_t native) {
    if (!native) return;
    com_ptr<ID3D12GraphicsCommandList> checked;
    if (!query_native(native, IID_ID3D12GraphicsCommandList, checked)) return;
    native = reinterpret_cast<std::uint64_t>(checked.p);
    // Destruction retires this recording but cannot complete GPU work. The
    // same fence/ownership requirements still protect every captured texture.
    std::lock_guard lock(mutex);
    const auto cookie = native_observer::get_recording_cookie(native);
    for_each_capture([&](auto &value) {
      retire_recording(value, cookie);
      // Never submitted, so its recorded read can no longer execute.
      if (value.pending_immediate == native) value.pending_immediate = 0;
    });
    if (auto owner = command_storage::acquire(checked.p, false)) {
      owner.life()->cookie.store(0, std::memory_order_release);
      owner->closed = true;
      owner->reported = false;
    }
  }
  void observe_queue(std::uint64_t native) {
    if (!requested.load() || !native) return;
    com_ptr<ID3D12CommandQueue> checked;
    if (!query_native(native, IID_ID3D12CommandQueue, checked)) { reason = status::unsupported_queue; return; }
    native = reinterpret_cast<std::uint64_t>(checked.p);
    native_observer::observe_queue(native);
    std::lock_guard lock(mutex);
    collect_retired();
    if (queue(native)) return;
    register_queue(reinterpret_cast<ID3D12CommandQueue *>(native), false);
  }
  void retire_queue(std::uint64_t native) {
    std::lock_guard lock(mutex);
    if (auto *owner = queue(native)) {
      owner->retiring = true;
      native = reinterpret_cast<std::uint64_t>(owner->queue.p);
      for_each_capture([&](auto &value) {
        if (!uses_queue(value, native)) return;
        invalidate(value, capture_failure::queue_retired);
        // The runtime's immediate list is destroyed with its queue; an
        // unsubmitted read can no longer execute.
        if (value.consumer_queue == native) value.pending_immediate = 0;
      });
    }
    collect_retired();
    // Its newest game execution goes with it, freeing that entry.
    forget_queue_execution(native);
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
    void set_release_wait(std::uint64_t ms) { release_wait_ms = ms ? ms : release_wait_bound_ms; }
  }
#endif
  bool release_device(std::uint64_t native, char *out, std::size_t size) {
    if (out && size) *out = 0;
    com_ptr<ID3D12Device> checked;
    if (!native || !query_native(native, IID_ID3D12Device, checked)) return false;
    const auto device = device_cookie(checked.p);
    if (!device) return false;
    // Its queue places take no further work, and each signals its fence after
    // everything submitted to it. Every copy into or read of a snapshot that
    // was submitted was submitted on a placed queue (an unplaced queue's copy
    // is never submitted, so never read; the game drained its own queues).
    std::array<std::pair<ID3D12Fence *, std::uint64_t>, queue_limit> fences{};
    unsigned waits = 0;
    bool finished = true;
    {
      std::lock_guard lock(mutex);
      // The add-on's own queue calls under the capture mutex are never game
      // calls to the order hooks (design finding 8).
      native_observer::suppression_scope suppress;
      for (auto &owner : queues) if (owner && owner->device_identity == device) {
        owner->retiring = true;
        owner->provider_established = false;
        const auto value = ++owner->next_fence;
        if (SUCCEEDED(owner->queue->Signal(owner->fence.p, value))) {
          owner->fence->AddRef();
          fences[waits++] = {owner->fence.p, value};
        } else if (SUCCEEDED(owner->device->GetDeviceRemovedReason())) finished = false; // A removed device runs nothing.
      }
    }
    // The device's queue-order records go with it (never dereferenced).
    native_observer::queue_order_forget_device(reinterpret_cast<std::uintptr_t>(checked.p));
    const auto started = GetTickCount64();
    const std::uint64_t bound = release_wait_ms;
    const HANDLE event = waits ? CreateEventW(nullptr, FALSE, FALSE, nullptr) : nullptr;
    for (unsigned i = 0; i != waits; ++i) {
      const auto [fence, value] = fences[i];
      // A removed device's fences read UINT64_MAX.
      if (finished && fence->GetCompletedValue() < value) {
        const auto elapsed = GetTickCount64() - started;
        finished = event && elapsed < bound && SUCCEEDED(fence->SetEventOnCompletion(value, event)) &&
          WaitForSingleObject(event, static_cast<DWORD>(bound - elapsed)) == WAIT_OBJECT_0;
      }
      fence->Release();
    }
    if (event) CloseHandle(event);
    const auto waited = GetTickCount64() - started;
    // Every object of the device leaves the capture owner: its snapshots and
    // queue places, and the resources of its sources and storage wherever
    // they are still held (an adapter's tag, a packet, a ticket), which are
    // released after the lock. Unfinished, they are kept for good instead.
    std::vector<std::shared_ptr<texture_reference>> storage_held;
    std::vector<source_ref> sources_held;
    std::vector<std::unique_ptr<queue_state>> places_held;
    std::vector<IUnknown *> released;
    unsigned snapshots = 0, places = 0, released_sources = 0, released_storage = 0;
    bool reserved = false;
    {
      std::lock_guard lock(mutex);
      // Everything below moves without allocating.
      try {
        constexpr std::size_t slot_count = slot_limit + diagnostic_slot_limit;
        storage_held.reserve(slot_count + storage.size());
        sources_held.reserve(2 * slot_count + source_limit);
        places_held.reserve(queue_limit);
        released.reserve(source_limit + 2 * storage.size());
        if (!finished) {
          if (!kept) kept = new kept_objects;
          kept->storage.reserve(kept->storage.size() + storage_held.capacity());
          kept->sources.reserve(kept->sources.size() + sources_held.capacity());
          kept->places.reserve(kept->places.size() + places_held.capacity());
        }
        reserved = true;
      } catch (...) {
        finished = false; // Out of memory: nothing of the device moved.
      }
      if (reserved) {
        std::array<std::uint64_t, queue_limit> natives{};
        for (auto &owner : queues) if (owner && owner->device_identity == device) {
          const auto queue = reinterpret_cast<std::uint64_t>(owner->queue.p);
          natives[places++] = queue;
          forget_queue_execution(queue);
          places_held.push_back(std::move(owner));
        }
        const auto of_device = [&](const slot &value) {
          return (value.texture && value.texture->device_identity == device) ||
            (value.published && value.published->device_identity == device) ||
            (value.metadata.source && value.metadata.source->device_identity == device) ||
            (value.auxiliary_destination && value.auxiliary_destination->device_identity == device) ||
            std::any_of(natives.begin(), natives.begin() + places, [&](auto queue) { return uses_queue(value, queue); });
        };
        const auto release_slot = [&](slot &value) {
          if (!of_device(value)) return;
          snapshots += value.id != 0;
          if (value.texture) storage_held.push_back(std::move(value.texture));
          if (value.published) storage_held.push_back(std::move(value.published));
          if (value.metadata.source) sources_held.push_back(std::move(value.metadata.source));
          if (value.auxiliary_destination) sources_held.push_back(std::move(value.auxiliary_destination));
          slot empty;
          // A record attempt may be allocating for this slot without the lock.
          empty.reservation = value.reservation; empty.reservation_tick = value.reservation_tick;
          empty.reserved_bytes = value.reserved_bytes; empty.reserved_live = value.reserved_live;
          value = std::move(empty);
        };
        for (auto &value : slots) release_slot(value);
        for (auto &value : diagnostic_slots) release_slot(value);
        for (auto &weak : sources) if (const auto value = weak.lock()) if (value->device_identity == device) {
          weak.reset();
          ++released_sources;
          if (!finished) { sources_held.push_back(value); continue; }
          value->released.store(true, std::memory_order_release);
          value->device = nullptr;
          released.push_back(std::exchange(value->resource.p, nullptr));
        }
        for (auto &weak : storage) if (const auto value = weak.lock()) if (value->device_identity == device) {
          weak.reset();
          ++released_storage;
          if (!finished) { storage_held.push_back(value); continue; }
          value->released.store(true, std::memory_order_release);
          value->device = nullptr;
          value->shader_resource = 0;
          released.push_back(std::exchange(value->resource.p, nullptr));
          released.push_back(std::exchange(value->views.p, nullptr));
        }
        if (!finished) {
          for (auto &value : storage_held) kept->storage.push_back(std::move(value));
          for (auto &value : sources_held) kept->sources.push_back(std::move(value));
          for (auto &value : places_held) kept->places.push_back(std::move(value));
        }
      }
    }
    // A device recreated over the same native object starts with a new identity.
    checked->SetPrivateData(sunshine_native_identity::device_guid, 0, nullptr);
    for (auto *value : released) if (value) value->Release();
    const auto u = [](std::uint64_t value) { return static_cast<unsigned long long>(value); };
    if (out && size) {
      if (!reserved)
        std::snprintf(out, size, "Sunshine capture: device %llu: out of memory at its release; its objects stay", u(device));
      else if (finished)
        std::snprintf(out, size, "Sunshine capture: released device %llu: queue_places=%u snapshots=%u sources=%u "
          "storage=%u gpu_wait_ms=%llu", u(device), places, snapshots, released_sources, released_storage, u(waited));
      else
        std::snprintf(out, size, "Sunshine capture: kept device %llu: its queues had not finished within %llu ms, so its "
          "queue_places=%u snapshots=%u sources=%u storage=%u are kept for good", u(device), u(waited), places, snapshots,
          released_sources, released_storage);
    }
    return finished;
  }
  namespace {
    // Idle snapshot storage kept for reuse beyond this is released.
    constexpr std::uint64_t idle_pixel_byte_limit = 256ull * 1024 * 1024;
    // Requires mutex. A reclaimable slot keeps only its allocation: the game
    // resource, its device and recording references go now rather than when
    // the slot is next reused (after a hitch many slots can be idle). Released
    // storage moves to discard, for release after the lock.
    void release_idle_slots(std::vector<std::shared_ptr<texture_reference>> &discard) {
      // An API namespace's newest nomination stays: selection reports its
      // failure or pending state.
      const auto head = [](const slot &value) {
        const auto &metadata = value.metadata;
        if (value.preservation_only || !metadata.epoch || !valid_provider(metadata.provider)) return false;
        const auto &current = evaluations[provider_index(metadata.provider)];
        return metadata.epoch == current.epoch && metadata.sequence == current.sequence && metadata.source_id == current.source_id;
      };
      std::uint64_t idle = 0;
      for (auto &value : slots) {
        // Predicted storage (U-C1b) whose submission made no copy-out: the
        // reader's queue executed it after all, or the copy keeps the Reset
        // rule. No list wrote it.
        if (value.id && value.predicted_storage && value.producer_submitted && !value.copied_out &&
            value.published.use_count() == 1) {
          discard.push_back(std::move(value.published));
          value.predicted_storage = false;
          copy_out_counters.released.fetch_add(1, std::memory_order_relaxed);
        }
        if (value.id && !head(value) && reclaimable(value)) {
          slot idle_slot;
          idle_slot.texture = std::move(value.texture);
          idle_slot.published = std::move(value.published);
          // A record attempt may be allocating for this slot without the lock.
          idle_slot.reservation = value.reservation; idle_slot.reservation_tick = value.reservation_tick;
          idle_slot.reserved_bytes = value.reserved_bytes; idle_slot.reserved_live = value.reserved_live;
          value = std::move(idle_slot);
        }
        if (value.id) continue;
        // Cached copy-out storage counts as the snapshot's (record_impl).
        for (auto *storage : {&value.texture, &value.published}) {
          if (!*storage) continue;
          if (idle + (*storage)->allocation_bytes > idle_pixel_byte_limit) discard.push_back(std::move(*storage));
          else idle += (*storage)->allocation_bytes;
        }
      }
    }
    // Cached colour/mask storage a live snapshot has not reused for this long.
    constexpr std::uint64_t idle_live_storage_ms = 2000;
    // Requires mutex. Its live budget is sized from the request, so idle storage
    // ages out instead of waiting for budget pressure. Released storage moves to
    // discard, for release after the lock.
    void release_idle_diagnostics(std::vector<std::shared_ptr<texture_reference>> &discard, std::uint64_t now) {
      for (auto &value : diagnostic_slots)
        if (!value.id && value.texture && !reserved_by_other(value, 0) && now - value.idle_tick >= idle_live_storage_ms)
          discard.push_back(std::move(value.texture));
    }
  }
  void poll() {
    if (requested.load()) native_observer::install_pending();
    std::vector<std::shared_ptr<texture_reference>> discard;
    std::lock_guard lock(mutex);
    collect_retired();
    try {
      release_idle_slots(discard);
      release_idle_diagnostics(discard, GetTickCount64());
    } catch (...) {}
    // discard is destroyed after the lock (declared before it).
  }

  void observe_provider(std::uint64_t native) {
    // API activity is discovery evidence only. The actual submitted capture's
    // queue is unknown here, and depth/metadata/evaluation may still be invalid.
    observe_command(native);
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
    frame_generation_on_tick.store(enabled_tick, std::memory_order_relaxed);
  }
  void frame_generation_frame(std::uint64_t epoch, std::uint32_t viewport, std::uint64_t enabled_tick) {
    if (!requested.load() || !epoch) return;
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
  void retire_source(provider_kind provider, std::uint64_t epoch, std::uint64_t source_id) {
    if (!valid_provider(provider) || !epoch || !source_id) return;
    std::lock_guard lock(mutex);
    const auto matches = [&](const input &value) {
      return value.provider == provider && value.epoch == epoch && value.source_id == source_id;
    };
    for (auto &value : slots) if (value.id && !value.preservation_only && matches(value.metadata)) {
      invalidate(value, capture_failure::source_retired);
    }
    for (auto &owner : queues) if (owner && owner->provider_established && owner->provider == provider &&
        owner->nominated_epoch == epoch && owner->source_id == source_id) {
      owner->provider_established = false;
      owner->source_id = owner->last_epoch = owner->last_sequence = owner->last_source_tick = owner->nominated_epoch = owner->present_capture = 0;
      // The next provider must submit a new frame after release. An otherwise
      // recent snapshot captured while the old provider owned presentation is
      // historical, and must not be resurrected by an FG Off/On transition.
      // IDs use pre-increment: serial is the last issued ID, not the next one.
      owner->admission_after = std::max(owner->admission_after, serial.load(std::memory_order_relaxed));
    }
    auto &current = evaluations[provider_index(provider)];
    for (auto &view : current.views) if (view.epoch == epoch && view.source_id == source_id) view = {};
    if (current.pending_nomination.epoch == epoch && current.pending_nomination.source_id == source_id)
      current.pending_nomination = {};
    if (current.epoch == epoch && current.source_id == source_id) {
      // Preserve the namespace's monotonic sequence watermark; releasing a
      // feature does not authorize replay of its earlier evaluation.
      current.source_id = 0;
      attempt_status[provider_index(provider)] = status::unavailable;
    }
  }

  source_ref retain_source(std::uint64_t native, record_diagnostic *diagnostic) {
    if (diagnostic) { *diagnostic = {}; diagnostic->resource = native; }
    const auto result = [&](source_ref value, status outcome, record_stage stage) {
      if (diagnostic) {
        diagnostic->result = outcome; diagnostic->stage = stage;
        if (value) {
          diagnostic->expected_device_identity = value->device_identity;
          diagnostic->width = static_cast<std::uint32_t>(value->desc.Width);
          diagnostic->height = value->desc.Height; diagnostic->format = value->desc.Format;
          diagnostic->flags = value->desc.Flags;
          diagnostic->dimension = value->desc.Dimension; diagnostic->mip_levels = value->desc.MipLevels;
          diagnostic->array_size = value->desc.DepthOrArraySize; diagnostic->samples = value->desc.SampleDesc.Count;
        }
      }
      return value;
    };
    if (!active()) return result({}, status::inactive, record_stage::inactive);
    if (!native) return result({}, status::malformed, record_stage::malformed_input);
    com_ptr<ID3D12Resource> checked;
    if (!query_native(native, IID_ID3D12Resource, checked)) return result({}, status::unsupported_resource, record_stage::source_interface);
    auto *resource = checked.p;
    const auto retained = [](std::uint64_t cookie) -> source_ref {
      if (cookie) for (const auto &weak : sources) if (auto value = weak.lock()) if (value->cookie == cookie) return value;
      return {};
    };
    {
      std::lock_guard lock(mutex);
      if (auto value = retained(source_cookie(resource))) return result(value, status::ready, record_stage::retained);
    }
    // A new source's device is read without the lock: under ReShade the
    // GetDevice hook and the proxy's Release take ReShade's device lock, which
    // it holds while it destroys a device (release_device takes this lock).
    auto value = std::make_shared<source_reference>();
    if (FAILED(resource->QueryInterface(__uuidof(ID3D12Resource), reinterpret_cast<void **>(value->resource.put()))))
      return result({}, status::unsupported_resource, record_stage::source_interface);
    ID3D12Device *device{};
    if (FAILED(resource->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void **>(&device))) || !device)
      return result({}, status::unsupported_queue, record_stage::source_device);
    // Not owned (source_reference): the game keeps its device while it
    // presents, and release_device drops this source before it is destroyed.
    value->device = device;
    value->device_identity = device_cookie(device, true);
    device->Release();
    value->desc = resource->GetDesc();
    if (diagnostic) {
      diagnostic->width = static_cast<std::uint32_t>(value->desc.Width);
      diagnostic->height = value->desc.Height; diagnostic->format = value->desc.Format;
      diagnostic->flags = value->desc.Flags;
      diagnostic->dimension = value->desc.Dimension; diagnostic->mip_levels = value->desc.MipLevels;
      diagnostic->array_size = value->desc.DepthOrArraySize; diagnostic->samples = value->desc.SampleDesc.Count;
    }
    if (!value->device_identity) { reason = status::unsupported_queue; return result({}, status::unsupported_queue, record_stage::device_identity); }
    // Retention proves object identity/lifetime, not capture capability. The
    // common opportunity recorder validates the actual copy description.
    value->cookie = retain_source_cookie(resource);
    if (!value->cookie) return result({}, status::unsupported_resource, record_stage::source_cookie);
    std::lock_guard lock(mutex);
    // Another call may have retained it meanwhile.
    if (auto existing = retained(value->cookie)) return result(existing, status::ready, record_stage::retained);
    auto entry = std::find_if(sources.begin(), sources.end(), [](const auto &value) { return value.expired(); });
    if (entry == sources.end()) {
      capacity_refused(source_table);
      reason = status::exhausted; return result({}, status::exhausted, record_stage::capacity);
    }
    // An adapter's source is watched for the enhanced-layout latch from now
    // on (proven_state.h); buffers and simultaneous-access textures keep no
    // layout between executions.
    if (value->desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER &&
        !(value->desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS)) {
      forget_destroyed();
      proven.watch(value->cookie);
    }
    *entry = value;
    return result(value, status::ready, record_stage::retained);
  }

  std::uint64_t source_present_generation(const source_ref &source) {
    return source ? source->present_generation.load(std::memory_order_acquire) : 0;
  }
  bool recorded_state(std::uint64_t native_command, const source_ref &source, bool *writing) {
    if (writing) *writing = false;
    if (!requested.load() || !native_command || !source) return false;
    com_ptr<ID3D12GraphicsCommandList> checked;
    if (!query_native(native_command, IID_ID3D12GraphicsCommandList, checked)) return false;
    const auto native = reinterpret_cast<std::uint64_t>(checked.p);
    if (!native_observer::states_ready(native)) return false;
    const auto cookie = native_observer::get_recording_cookie(native);
    std::lock_guard lock(mutex);
    auto owner = command(native, cookie, false);
    const auto *entry = owner && !owner->invalid ? state(*owner, source->cookie, false) : nullptr;
    if (entry && writing) *writing = copy_state::writes(observation_of(entry));
    return entry != nullptr;
  }
  std::uint64_t game_executions() { return game_execution_count.load(std::memory_order_acquire); }
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
  bool source_extent(const source_ref &source, std::uint32_t &width, std::uint32_t &height) {
    width = height = 0;
    if (!source || !source->desc.Width || source->desc.Width > UINT32_MAX) return false;
    width = static_cast<std::uint32_t>(source->desc.Width); height = source->desc.Height;
    return true;
  }
  void present(std::uint64_t native) {
    if (!requested.load() || !native) return;
    std::lock_guard lock(mutex);
    const auto *owner = queue(native);
    if (!owner) return;
    const auto device = device_cookie(owner->device.p);
    if (!device) return;
    for (const auto &weak : sources) if (const auto source = weak.lock())
      if (source->device_identity == device) ++source->present_generation;
  }

  static void begin_evaluation_locked(std::uint64_t epoch, std::uint64_t sequence, std::uint32_t viewport,
      provider_kind provider, std::uint64_t source_id, std::uint64_t superseded_source_id) {
    if (!requested.load() || !valid_provider(provider)) return;
    auto &current = evaluations[provider_index(provider)];
    if (epoch != current.epoch) { current = {}; current.epoch = epoch; }
    if (sequence <= current.sequence) return;
    current.sequence = sequence;
    current.source_id = source_id;
    current.pending_nomination = {};
    attempt_status[provider_index(provider)] = status::unavailable;
    // An explicit adapter role change (SR -> FG) replaces one logical source
    // for this same viewport. Other viewports remain independently ambiguous;
    // this changes neither captured GPU lifetimes nor provider ownership.
    if (superseded_source_id != UINT64_MAX && superseded_source_id != source_id)
      for (auto &value : current.views)
        if (value.epoch == epoch && value.viewport == viewport && value.source_id == superseded_source_id) value = {};
    const auto now = GetTickCount64();
    for (auto &value : current.views) if (value.epoch == epoch && value.viewport == viewport && value.source_id == source_id) {
      // A new call does not end a run of rejected ones; only its result can.
      value.sequence = sequence; value.tick = now; return;
    }
    auto oldest = std::min_element(current.views.begin(), current.views.end(), [](const auto &a, const auto &b) { return a.sequence < b.sequence; });
    *oldest = {epoch, sequence, now, source_id, viewport};
  }

  void begin_evaluation(std::uint64_t epoch, std::uint64_t sequence, std::uint32_t viewport,
      provider_kind provider, std::uint64_t source_id, std::uint64_t superseded_source_id) {
    std::lock_guard lock(mutex);
    begin_evaluation_locked(epoch, sequence, viewport, provider, source_id, superseded_source_id);
  }

  namespace {
    // Requires mutex. value, its provider's newest evaluation, is a nomination
    // in progress (pending_nomination) until its copy is recorded or refused
    // (end_nomination). Every evaluation's, SR's as well as FG's: a Present
    // meanwhile, another thread's, holds instead of reading that attempt as
    // refused, which moved acquire's gap watermark and so revoked the
    // source's earlier snapshots still in flight on their queue (Assassin's
    // Creed Shadows 10-25: with FG on, FG Presents landed while the SR
    // evaluate allocated storage for each dynamic-resolution size, and each
    // revoked SR depth for a frame or more). false: value is not the newest
    // evaluation, or (unless renewed) its nomination is already in progress
    // and its own caller ends it.
    bool mark_pending(evaluation_namespace &current, const evaluation_head &nomination, bool renewed) {
      if (current.epoch != nomination.epoch || current.sequence != nomination.sequence ||
          current.source_id != nomination.source_id || !nomination.epoch || !nomination.sequence ||
          (!renewed && same_nomination(current.pending_nomination, nomination)))
        return false;
      current.pending_nomination = nomination;
      current.pending_nomination.tick = GetTickCount64();
      return true;
    }
    // An adapter's input names its retained source and resource as well.
    bool mark_pending(evaluation_namespace &current, const input &value, bool renewed) {
      return value.source && value.resource.native && mark_pending(current, nomination_of(value), renewed);
    }
  }
  SUNSHINE_CAPTURE_STATIC bool begin_nomination(const input &value, std::uint64_t superseded_source_id) {
    std::lock_guard lock(mutex);
    begin_evaluation_locked(value.epoch, value.sequence, value.viewport, value.provider, value.source_id, superseded_source_id);
    if (!requested.load() || !valid_provider(value.provider)) return false;
    return mark_pending(evaluations[provider_index(value.provider)], value, true);
  }
  void evaluation_nomination::begin(std::uint64_t epoch, std::uint64_t sequence, std::uint32_t viewport,
      provider_kind provider, std::uint64_t source_id, std::uint64_t superseded_source_id) {
    end();
    std::lock_guard lock(mutex);
    begin_evaluation_locked(epoch, sequence, viewport, provider, source_id, superseded_source_id);
    if (!requested.load() || !valid_provider(provider)) return;
    const evaluation_head nomination{epoch, sequence, 0, source_id, viewport};
    if (mark_pending(evaluations[provider_index(provider)], nomination, true))
      nominated = {provider, epoch, sequence, source_id, viewport, true};
  }
  void evaluation_nomination::end() {
    if (!std::exchange(nominated.pending, false)) return;
    end_nomination(nominated.provider, {nominated.epoch, nominated.sequence, 0, nominated.source_id, nominated.viewport});
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
    bool fail_copy_out_allocation() {
      for (auto count = copy_out_allocation_failures.load(); count;)
        if (copy_out_allocation_failures.compare_exchange_weak(count, count - 1)) return true;
      return false;
    }
  }
#endif
  SUNSHINE_CAPTURE_STATIC bool source_lifetime_current(const input &value, std::uint64_t current_generation) {
    // A copy recorded synchronously inside the authenticated call that
    // supplied the resource (a tag call, or an evaluation's own local input)
    // is current by construction, for OnlyValidNow and UntilPresent alike: an
    // independent FG Present of an earlier frame cannot end this call's
    // lifetime. With frame generation such Presents land between reading the
    // input and recording its copy on a fraction of real frames, more often
    // while new storage is allocated without the lock (Assassin's Creed
    // Shadows 10-24: its SR evaluation's local depth input, each dynamic
    // resolution change). A later use of the tag's own frame (own_frame) is current for
    // the same reason: a Present since the tag was an earlier frame's (the
    // matrix's S7). Any other later use, and a delayed preservation copy,
    // still require an unchanged generation.
    if (value.at_tag_call || value.own_frame ||
        (value.valid_until == sunshine_scene_depth::lifetime::at_call && value.force_snapshot))
      return true;
    return value.source_present_generation && value.source_present_generation == current_generation;
  }

  // proven_queue: the queue that executes native, for a last_proven proof.
  static std::uint64_t record_impl(std::uint64_t native, const input &value, record_diagnostic *diagnostic,
      bool preservation_only, preservation_ticket *preserved = nullptr, bool nominate_source = false,
      bool diagnostic_copy = false, bool reusable_storage = false,
      texture_state_policy state_policy = texture_state_policy::source_contract, std::uint64_t proven_queue = 0) {
    bool valid_nomination = false;
    std::uint64_t nomination_command{}, nomination_source{};
    recording_ref nomination_recording;
    if (diagnostic) {
      *diagnostic = {};
      diagnostic->command = native; diagnostic->resource = value.resource.native;
      diagnostic->expected_generation = value.source_present_generation;
      diagnostic->native_state = value.native_state;
      if (value.source) {
        diagnostic->expected_device_identity = value.source->device_identity;
        diagnostic->width = static_cast<std::uint32_t>(value.source->desc.Width);
        diagnostic->height = value.source->desc.Height; diagnostic->format = value.source->desc.Format;
        diagnostic->flags = value.source->desc.Flags;
        diagnostic->dimension = value.source->desc.Dimension; diagnostic->mip_levels = value.source->desc.MipLevels;
        diagnostic->array_size = value.source->desc.DepthOrArraySize; diagnostic->samples = value.source->desc.SampleDesc.Count;
      }
    }
    const auto reject = [&](status result, record_stage stage) -> std::uint64_t {
      if (diagnostic) { diagnostic->result = result; diagnostic->stage = stage; }
      if (!preservation_only && !value.pixels_required) set_attempt(value.provider, result);
      if (nominate_source && valid_nomination && !value.pixels_required) {
        // Pixel capability must not determine source authority. This carries
        // the validated source through original evaluation completion and real
        // submission without fabricating an owned copy or another source.
        const auto ticket = record_nomination(value, nomination_source, nomination_command, nomination_recording);
        if (ticket) {
          for (auto &entry : slots) if (entry.id == ticket) {
            entry.nomination_only = entry.source_nominated = true;
            entry.pixel_failure = result; break;
          }
        }
        return ticket;
      }
      return 0;
    };
    if (!requested.load()) return reject(status::malformed, record_stage::inactive);
    const bool prefer_observed_recording = state_policy == texture_state_policy::prefer_observed_recording;
    if (state_policy != texture_state_policy::source_contract &&
        (!prefer_observed_recording || !preservation_only || !diagnostic_copy ||
          value.provider != provider_kind::streamline || !value.force_snapshot ||
          value.valid_until != sunshine_scene_depth::lifetime::at_call))
      return reject(status::malformed, record_stage::malformed_input);
    if (!native) return reject(status::malformed, record_stage::malformed_input);
    // A source released with its device (release_device): its resource, which
    // value.resource names, may be gone.
    if (!value.source || value.source->released.load(std::memory_order_acquire))
      return reject(status::malformed, record_stage::missing_source);
    if ((!preservation_only && (!valid_provider(value.provider) || !valid_projection(value.projection) ||
        !value.epoch || !value.sequence)) || !value.resource.native)
      return reject(status::malformed, record_stage::malformed_input);
    if (value.valid_until != sunshine_scene_depth::lifetime::until_present &&
        value.valid_until != sunshine_scene_depth::lifetime::until_evaluation &&
        !(value.valid_until == sunshine_scene_depth::lifetime::at_call && value.force_snapshot))
      return reject(status::unsupported_lifetime, record_stage::unsupported_lifetime);
    // Every capture copies this call's contents and needs a state proof.
    const auto state_rule = copy_state::select(value.proof, state_policy);
    if (state_rule == copy_state::rule::none)
      return reject(status::unsupported_state, record_stage::unsupported_proof);
    com_ptr<ID3D12GraphicsCommandList> checked;
    if (!query_native(native, IID_ID3D12GraphicsCommandList, checked))
      return reject(status::unsupported_queue, record_stage::command_interface);
    com_ptr<ID3D12Resource> checked_source;
    if (!query_native(value.resource.native, IID_ID3D12Resource, checked_source) ||
        source_cookie(checked_source.p) != value.source->cookie)
      return reject(status::unsupported_resource, record_stage::source_identity);
    native = reinterpret_cast<std::uint64_t>(checked.p);
    if (diagnostic) diagnostic->command = native;
    observe_command(native);
    // The barrier hooks supply the list's resource states only when they see it.
    const bool states_observed = native_observer::states_ready(native);
    // A proven state holds for a queue whose executions the hooks report. It
    // supplies a last_proven proof, and checks the declaration of a copy
    // deferred to its frame's Present (input::present_queue) as an
    // observation would: one that disagrees refuses the copy.
    com_ptr<ID3D12CommandQueue> present_queue;
    if (!proven_queue && query_native(value.present_queue, IID_ID3D12CommandQueue, present_queue))
      proven_queue = reinterpret_cast<std::uint64_t>(present_queue.p); // As the submission hook reports it.
    const bool proven_basis_allowed = (value.proof == sunshine_scene_depth::state_proof::last_proven ||
        (value.present_queue && value.proof == sunshine_scene_depth::state_proof::declared)) && states_observed &&
      proven_queue && native_observer::queue_ready(proven_queue);
    auto *list = checked.p;
    const auto command_type = list->GetType();
    if (diagnostic) diagnostic->command_type = command_type;
    // NGX and Streamline evaluate on any non-copy queue (the matrix's S22 and
    // N7), so a copy is recorded on a DIRECT or COMPUTE list, from a basis
    // that list may name (copy_state::list_basis).
    const bool on_compute = command_type == D3D12_COMMAND_LIST_TYPE_COMPUTE;
    if (command_type != D3D12_COMMAND_LIST_TYPE_DIRECT && !on_compute)
      return reject(status::unsupported_queue, record_stage::command_type);
    com_ptr<ID3D12Device> list_device;
    if (FAILED(list->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void **>(list_device.put()))))
      return reject(status::unsupported_queue, record_stage::command_device);
    const auto device_identity = device_cookie(list_device.p);
    if (diagnostic) diagnostic->device_identity = device_identity;
    if (device_identity != value.source->device_identity) return reject(status::unsupported_queue, record_stage::device_identity);
    const auto cookie = native_observer::get_recording_cookie(native);
    if (diagnostic) diagnostic->recording_cookie = cookie;
    // Game threads never hold the capture lock while the driver allocates.
    // An attempt that needs new storage reserves a slot (and its diagnostic
    // budget) under the lock, allocates unlocked, then repeats every
    // admission check from the start under the lock before it publishes. A
    // reservation is only a preference: publication never trusts it.
    std::shared_ptr<texture_reference> prepared, prepared_published;
    // Copy-out storage this call failed to allocate or register.
    bool copy_out_refused = false;
    std::uint64_t reservation{};
    // Storage replaced under the lock is released after it (Release of a
    // large committed resource can take milliseconds).
    std::vector<std::shared_ptr<texture_reference>> discarded;
    for (unsigned attempt = 0;; ++attempt) {
      std::unique_lock lock(mutex);
      valid_nomination = false;
      // Every exit below holds the lock, and a reservation never outlives the
      // call: only the retry after an unlocked allocation keeps it.
      struct reservation_release {
        std::uint64_t &token;
        bool keep{};
        ~reservation_release() { if (token && !keep) clear_reservation(std::exchange(token, 0)); }
      } release_on_exit{reservation};
      // A compute list's copy is accepted only once its queue holds a place
      // (register_compute_producer): with none left for it and no compute
      // queue holding one, no GPU copy is recorded that could never be read.
      if (on_compute && compute_unplaced && !compute_producer_placed()) {
        if (!free_place()) return reject(status::exhausted, record_stage::capacity);
        compute_unplaced = false;
      }
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
      if (nominate_source && !preservation_only && !diagnostic_copy &&
          present_before_next_copy.exchange(false, std::memory_order_acq_rel))
        ++value.source->present_generation;
#endif
      if (!value.source->resource.p) return reject(status::malformed, record_stage::missing_source);
      const auto current_generation = source_present_generation(value.source);
      if (diagnostic) diagnostic->current_generation = current_generation;
      if (!source_lifetime_current(value, current_generation))
        return reject(status::unsupported_lifetime, record_stage::source_lifetime);
      auto owner = command(native, cookie, true);
      // ReShade's lifecycle decides coverage: the recording is open outside a
      // render pass.
      const auto view = lifecycle_view(owner ? &*owner : nullptr);
      if (!attempt) count_admission(view, states_observed);
      if (view == list_view::unknown) return reject(status::unavailable, record_stage::observer_coverage);
      if (diagnostic) {
        diagnostic->loss = owner->invalidation; diagnostic->recording_closed = owner->closed;
        diagnostic->recording_invalid = owner->invalid; diagnostic->render_pass = owner->render_pass;
        diagnostic->native_barrier_calls = owner->native_barrier_calls;
        diagnostic->native_transition_count = owner->native_transition_count;
        diagnostic->tracked_source_count = owner->states.size();
        diagnostic->last_barrier_command = owner->last_barrier_command;
        diagnostic->source_cookie = value.source->cookie;
      }
      if (view == list_view::closed) return reject(status::unavailable, record_stage::recording_closed);
      if (view == list_view::pass) return reject(status::unavailable, record_stage::recording_render_pass);
      const auto &desc = value.source->desc;
      copy_region region;
      const auto source_status = source_region(desc, value.resource, region);
      if (source_status != status::ready) return reject(source_status, record_stage::resource_region);
      valid_nomination = nominate_source;
      nomination_command = cookie; nomination_source = reinterpret_cast<std::uint64_t>(checked_source.p);
      nomination_recording = owner.life();
      const auto region_status = diagnostic_copy ?
        (diagnostic_description(desc, value.resource.native_format) ? source_region(desc, value.resource, region) :
          status::unsupported_resource) :
        capture_region(desc, value.resource, region);
      if (region_status != status::ready) return reject(region_status, record_stage::resource_region);
      // A depth snapshot is read through one view, resolved now; a linear
      // R16_TYPELESS without a declared format has none.
      const auto snapshot_view = diagnostic_copy ? DXGI_FORMAT_UNKNOWN : depth_view(desc.Format, value);
      if (!diagnostic_copy && snapshot_view == DXGI_FORMAT_UNKNOWN)
        return reject(status::unsupported_resource, record_stage::resource_region);
      if (owner->invalid) return reject(status::unavailable, record_stage::recording_invalid);
      // Unobserved lists (refused or moved method tables) use the declared state.
      const auto *observed = states_observed ? state(*owner, value.source->cookie, false) : nullptr;
      if (diagnostic && observed) {
        diagnostic->observed = true; diagnostic->observed_state = observed->value; diagnostic->blocked = observed->blocked;
        diagnostic->pending_split = observed->pending_split;
        diagnostic->observed_enhanced = observed->enhanced;
        diagnostic->observed_layout = observed->layout; diagnostic->observed_access = observed->access;
      }
      // A last proven state stands in for the observation this recording
      // lacks: the source's state proven on the copy's queue, unless this
      // recording aliased it before the copy (proven_state.h).
      sunshine_proven_state::state proven_basis;
      if (proven_basis_allowed && !observed && !owner->wildcard_alias && !aliased_after(*owner, value.source->cookie, 0))
        proven_basis = proven.query(value.source->cookie, proven_queue, !enhanced_barriers_seen.load(std::memory_order_relaxed));
      // Otherwise a source an enhanced barrier on an earlier executed list
      // latched (proven_state.h) takes no legacy barrier from its claim: the
      // claim is checked against its proof from that list (copy_state's
      // earlier observation, the matrix's D6 and N8). That proof stands only
      // as one above would: the hooks see this recording, which did not alias
      // it before the copy, and it was made on the copy's queue when that is
      // known; else the copy is refused.
      sunshine_proven_state::latch earlier;
      if (!observed && !proven_basis) {
        earlier = proven.latched(value.source->cookie);
        if (!states_observed || owner->wildcard_alias || aliased_after(*owner, value.source->cookie, 0) ||
            (proven_queue && earlier.proven.queue != proven_queue))
          earlier.proven = {};
      }
      const auto &proof = proven_basis ? proven_basis : earlier.proven;
      if (diagnostic) diagnostic->earlier_layout = earlier.latched;
      if (diagnostic && proof) {
        diagnostic->used_proven_state = true; diagnostic->observed_state = proof.value;
        diagnostic->observed_enhanced = proof.enhanced;
        diagnostic->observed_layout = proof.layout; diagnostic->observed_access = proof.access;
      }
      auto evidence = proof ? copy_state::observation{true, true, false, proof.value, false,
        proof.enhanced, proof.layout, proof.access} : observation_of(observed);
      evidence.earlier = earlier.latched;
      const auto chosen = copy_state::resolve(state_rule, value.native_state, evidence, desc.Flags);
      if (!chosen.admitted) {
        if (earlier.latched) proven_counters.earlier_refused.fetch_add(1, std::memory_order_relaxed);
        return reject(chosen.result, chosen.stage);
      }
      // A compute list's copy leaves and restores only a state or layout that
      // list may name, a direct list's none of a compute queue's own layouts
      // (S22b).
      if (!copy_state::list_basis(chosen, on_compute))
        return reject(status::unsupported_state, record_stage::unsupported_state);
      // The D3D12 contract forbids a legacy barrier on a texture in an
      // enhanced layout: that copy needs this list's CommandList7 Barrier.
      com_ptr<ID3D12GraphicsCommandList7> enhanced_list;
      if (chosen.enhanced && !query_native(native, IID_ID3D12GraphicsCommandList7, enhanced_list))
        return reject(status::unsupported_queue, record_stage::command_interface);
      if (diagnostic) diagnostic->enhanced_copy = chosen.enhanced;
      const auto before = chosen.state;
      const auto basis = proof ? state_basis::proven : copy_basis(observed, chosen);
      if (diagnostic) {
        diagnostic->copy_state = before; diagnostic->copy_state_known = true;
        diagnostic->used_observed_state = chosen.used_observed;
        diagnostic->used_contract_state = chosen.used_contract;
      }
      // Multiple legal clear boundaries in one unsubmitted recording may replace
      // the same source snapshot before anyone reads it. Old tickets are retired
      // by ID; their CPU leases alone do not represent recorded GPU reads.
      if (diagnostic_copy) collect_diagnostics();
      auto *pixel_begin = diagnostic_copy ? diagnostic_slots.data() : slots.data();
      auto *pixel_end = pixel_begin + (diagnostic_copy ? diagnostic_slot_limit : pixel_slot_limit);
      const auto texture_format = diagnostic_copy ? diagnostic_storage_format(desc.Format, value.resource.native_format) :
        typeless(desc.Format);
      // A shared copy is opened by the host: only formats its table knows.
      if (diagnostic_copy && !reusable_storage && !game3d_debug::pixel_bytes(static_cast<unsigned>(texture_format)))
        return reject(status::unsupported_resource, record_stage::resource_region);
      const auto compatible_storage = [&](const auto &texture) {
        if (!texture || texture->device_identity != value.source->device_identity) return false;
        if (diagnostic_copy && bool(texture->shared_handle) == reusable_storage) return false;
        // A compute list cannot leave the pixel-shader state a depth
        // snapshot rests in; a direct list may use storage of either rest.
        if (on_compute && texture->rest != D3D12_RESOURCE_STATE_COMMON) return false;
        const auto existing = texture->resource->GetDesc();
        return existing.Width == region.width && existing.Height == region.height && existing.Format == texture_format &&
          (diagnostic_copy || texture->view_format == snapshot_view);
      };
      // Another attempt's live reservation is not available to this one.
      const auto free_for_us = [&](const slot &entry) { return !reserved_by_other(entry, reservation); };
      auto *found = preservation_only && !diagnostic_copy ? std::find_if(pixel_begin, pixel_end, [&](const auto &entry) {
        return reusable_preserved(entry, value, cookie);
      }) : pixel_end;
      // Prefer the slot this attempt reserved, if it is still empty and free.
      if (found == pixel_end && reservation) found = std::find_if(pixel_begin, pixel_end, [&](auto &entry) {
        return entry.reservation == reservation && !entry.id;
      });
      if (found == pixel_end && diagnostic_copy && reusable_storage) found = std::find_if(pixel_begin, pixel_end, [&](const auto &entry) {
        return !entry.id && free_for_us(entry) && compatible_storage(entry.texture);
      });
      // Do not overwrite the only newly completed API snapshot with its pending
      // successor before the effects queue can consume it. This borrows an
      // existing pool slot; no extra owner, queue, or resource copy is introduced.
      const auto *completed = nominate_source ? completed_snapshot(value, GetTickCount64()) : nullptr;
      // A Dump 3D copy (never cached) takes only a slot without live storage:
      // a live snapshot reuses its cache every frame. A full pool is exhausted.
      const bool dump_copy = diagnostic_copy && !reusable_storage;
      if (found == pixel_end) found = std::find_if(pixel_begin, pixel_end, [&](auto &entry) {
        return &entry != completed && free_for_us(entry) && reclaimable(entry) &&
          !(dump_copy && entry.texture && live_storage(entry));
      });
      if (found == pixel_end) {
        capacity_refused(diagnostic_copy ? auxiliary_pool : snapshot_pool);
        return reject(status::exhausted, record_stage::capacity);
      }
      auto texture = found->texture;
      // Copy-out storage (slot::published) for a copy of a source another
      // queue reads (copy_out): the snapshot's shape, view and rest. Each
      // comes from the slot's cache, else from an earlier attempt's
      // allocation (prepared), else is allocated now. U-C1b: the queue a
      // recording executes on is known only at its submission, so a copy
      // also takes it on a COMPUTE list, which never executes on the
      // presenting queue that reads depth, and on a list not known to
      // execute on the reader's queue: never executed yet, or last executed
      // on another queue, as the render queue's lists are once FG On moved
      // Present to DLSS-G's queue (the idle sweep releases it when the
      // submission is the reader's after all). A list never executed is
      // predicted to run where the newest DIRECT-list copy ran
      // (last_direct_copy_queue): with FG Off, the reader's own queue, so a
      // game that pools or creates its lists allocates nothing for them
      // (until 10-27 each new list's first copy allocated copy-out storage on
      // the game's thread, released unused on the Present thread). Storage
      // this attempt could not allocate is not sought again (copy_out
      // failed): the copy keeps the Reset rule. A list predicted to run on
      // the reader's queue takes it too while FG may move Present off that
      // queue (reader_may_move: the copies of FG's first Present).
      const auto record_tick = GetTickCount64();
      const bool read_elsewhere_before = read_elsewhere_recently(*value.source, record_tick);
      const bool predicted = !read_elsewhere_before && !on_compute && reader_live() &&
        ((owner->executed_queue ? owner->executed_queue != reader : last_direct_copy_queue != reader) ||
          reader_may_move(record_tick));
      const bool wants_published = !diagnostic_copy && !preservation_only && !copy_out_refused &&
        (read_elsewhere_before || on_compute || predicted);
      auto published = wants_published ? found->published : nullptr;
      // New storage's rest (texture_reference::rest).
      const auto resting_state = diagnostic_copy || on_compute ? D3D12_RESOURCE_STATE_COMMON : sampled_state;
      const bool texture_cached = compatible_storage(texture);
      const bool published_cached = !wants_published || compatible_storage(published);
      const bool texture_prepared = !texture_cached && prepared && compatible_storage(prepared);
      const bool published_prepared = !published_cached && prepared_published && compatible_storage(prepared_published);
      const bool need_texture = !texture_cached && !texture_prepared;
      const bool need_published = !published_cached && !published_prepared;
      if (!need_texture && !need_published) {
        if (texture_prepared) {
          // Storage allocated by an earlier attempt. Its budget was reserved on
          // the slot it reserved; a different slot must admit it again.
          if (diagnostic_copy && found->reservation != reservation &&
              !reserve_diagnostic_bytes(diagnostic_slots, found, prepared->allocation_bytes, reusable_storage, reservation, &discarded))
            return reject(status::exhausted, record_stage::capacity);
          texture = std::move(prepared);
        }
        if (published_prepared) published = std::move(prepared_published);
      } else {
        // Bounded: a second allocation means the shape changed meanwhile.
        if (attempt >= 2) return reject(status::exhausted, record_stage::capacity);
        storage_plan plan;
        plan.device = value.source->device;
        plan.device_identity = value.source->device_identity;
        auto &destination = plan.desc;
        destination = desc; destination.Format = texture_format; destination.Flags = D3D12_RESOURCE_FLAG_NONE;
        destination.Width = region.width; destination.Height = region.height;
        destination.MipLevels = 1; destination.DepthOrArraySize = 1; // Subresource 0 only.
        plan.state = resting_state;
        if (diagnostic_copy) {
          destination.Alignment = 0;
          destination.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
          // D3D11's host OpenSharedResource1 requires the D3D12 shared color
          // allocation to be RT-capable. This is a private-copy bind capability,
          // never a requirement on the application's mask/input allocation.
          D3D12_FEATURE_DATA_FORMAT_SUPPORT support{texture_format};
          if (FAILED(plan.device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))) ||
              !(support.Support1 & D3D12_FORMAT_SUPPORT1_RENDER_TARGET))
            return reject(status::unsupported_resource, record_stage::resource_region);
          destination.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
          const auto allocation = plan.device->GetResourceAllocationInfo(0, 1, &destination);
          if (!reserve_diagnostic_bytes(diagnostic_slots, found, allocation.SizeInBytes, reusable_storage, reservation, &discarded))
            return reject(status::exhausted, record_stage::capacity);
          plan.allocation_bytes = allocation.SizeInBytes;
          plan.shared = !reusable_storage;
          // Local snapshots can be bound directly (lease_local_view).
          plan.descriptors = reusable_storage;
          // The replacement is retired too. Release its incompatible storage
          // before allocation so even the replacement peak respects the budget.
          discarded.push_back(std::move(found->texture));
        } else {
          plan.descriptors = true;
          plan.view = snapshot_view;
          // Sizes the idle-storage budget (release_idle_slots).
          plan.allocation_bytes = plan.device->GetResourceAllocationInfo(0, 1, &destination).SizeInBytes;
        }
        clear_reservation(reservation);
        reservation = ++reservation_serial;
        found->reservation = reservation; found->reservation_tick = GetTickCount64();
        found->reserved_bytes = plan.allocation_bytes; found->reserved_live = diagnostic_copy && reusable_storage;
        lock.unlock();
        discarded.clear();
        if (need_texture) prepared.reset();
        if (need_published) prepared_published.reset();
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
        if (auto *hook = testing::allocation_hook.exchange(nullptr)) hook(testing::allocation_context.load());
#endif
        auto stage = need_texture ? allocate_storage(plan, prepared) : record_stage::recorded;
        // Copy-out storage is optional: a copy without it keeps the Reset
        // rule, as before U-C1b, and its submission records its list's
        // queue, so a list never executed is not predicted again.
        bool published_failed = false;
        if (stage == record_stage::recorded && need_published) {
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
          if (testing::fail_copy_out_allocation()) published_failed = true;
          else
#endif
          published_failed = allocate_storage(plan, prepared_published) != record_stage::recorded;
        }
        lock.lock();
        if (stage != record_stage::recorded) return reject(status::failed, stage);
        // Released with its device whoever holds it (release_device).
        if (need_texture && !register_storage(prepared)) return reject(status::failed, record_stage::texture_allocation);
        if (need_published && !published_failed && !register_storage(prepared_published)) published_failed = true;
        if (published_failed) {
          discarded.push_back(std::move(prepared_published));
          copy_out_refused = true;
          copy_out_counters.failed.fetch_add(1, std::memory_order_relaxed);
        }
        release_on_exit.keep = true;
        continue;
      }
      // Publication: this attempt passed every check above with the lock held.
      discarded.push_back(std::move(found->texture));
      discarded.push_back(std::move(found->published));
      *found = {};
      if (reservation) clear_reservation(std::exchange(reservation, 0));
      found->texture = std::move(texture); found->published = std::move(published); found->metadata = value; found->metadata.native_state = before;
      found->predicted_storage = predicted && found->published;
      if (found->predicted_storage) copy_out_counters.predicted.fetch_add(1, std::memory_order_relaxed);
      found->id = ++serial; found->command = cookie; found->producer_recording = owner.life();
      found->basis = basis; found->compute_list = on_compute;
      compute_copies_recorded |= on_compute;
      if (proof) {
        found->proven_source = value.source->cookie; found->proven_version = proof.version;
        found->proven_queue = proven_basis ? proven_queue : proof.queue;
      }
      if (earlier.latched) proven_counters.earlier_copied.fetch_add(1, std::memory_order_relaxed);
      found->preservation_only = preservation_only;
      found->diagnostic_only = diagnostic_copy;
      found->diagnostic_reusable = reusable_storage;
      if (diagnostic_copy) diagnostic_slots_active = true;
      found->source_nominated = nominate_source;
      found->finished = found->success = preservation_only && !diagnostic_copy;
      native_observer::suppression_scope suppress;
      D3D12_RESOURCE_BARRIER transitions[2]{};
      transitions[0].Type = transitions[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      transitions[0].Transition = {value.source->resource.p, 0, static_cast<D3D12_RESOURCE_STATES>(before), D3D12_RESOURCE_STATE_COPY_SOURCE};
      transitions[1].Transition = {found->texture->resource.p, 0, found->texture->rest, D3D12_RESOURCE_STATE_COPY_DEST};
      // An enhanced source leaves its exact layout and access for the copy
      // after all prior work (subresource 0: the zero range is index form).
      D3D12_TEXTURE_BARRIER source_layout{};
      D3D12_BARRIER_GROUP source_group{};
      if (chosen.enhanced) {
        source_layout.SyncBefore = D3D12_BARRIER_SYNC_ALL; source_layout.SyncAfter = D3D12_BARRIER_SYNC_COPY;
        source_layout.AccessBefore = static_cast<D3D12_BARRIER_ACCESS>(chosen.access);
        source_layout.AccessAfter = D3D12_BARRIER_ACCESS_COPY_SOURCE;
        source_layout.LayoutBefore = static_cast<D3D12_BARRIER_LAYOUT>(chosen.layout);
        source_layout.LayoutAfter = static_cast<D3D12_BARRIER_LAYOUT>(copy_state::layouts::copy_source(chosen.layout));
        source_layout.pResource = value.source->resource.p;
        source_group.Type = D3D12_BARRIER_TYPE_TEXTURE; source_group.NumBarriers = 1; source_group.pTextureBarriers = &source_layout;
        list->ResourceBarrier(1, &transitions[1]);
        enhanced_list->Barrier(1, &source_group);
      } else if (before == D3D12_RESOURCE_STATE_COPY_SOURCE) list->ResourceBarrier(1, &transitions[1]);
      else list->ResourceBarrier(2, transitions);
      D3D12_TEXTURE_COPY_LOCATION source{}, destination{};
      source.pResource = value.source->resource.p; source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      destination.pResource = found->texture->resource.p; destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      // Subresource zero is mip zero/array zero/plane zero. capture_region rejects
      // multi-mip, arrays and MSAA. Packed stencil planes remain untouched.
      list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
      std::swap(transitions[0].Transition.StateBefore, transitions[0].Transition.StateAfter);
      std::swap(transitions[1].Transition.StateBefore, transitions[1].Transition.StateAfter);
      if (chosen.enhanced) {
        // Restore the exact layout and access. SyncAfter ALL makes every later
        // command wait for the copy: the enhanced form of the legacy write
        // boundary below, so a later game write cannot overlap the read.
        std::swap(source_layout.AccessBefore, source_layout.AccessAfter);
        std::swap(source_layout.LayoutBefore, source_layout.LayoutAfter);
        source_layout.SyncBefore = D3D12_BARRIER_SYNC_COPY; source_layout.SyncAfter = D3D12_BARRIER_SYNC_ALL;
        enhanced_list->Barrier(1, &source_group);
      } else if (before != D3D12_RESOURCE_STATE_COPY_SOURCE && !(before & write_states)) {
        // The asynchronous copy/reuse regression exposes partial later writes
        // when restoring only a read state here. Complete the injected source
        // read at an explicit write-state boundary before restoring the game's
        // exact state. This adds ordering, never a write to the game resource.
        auto completion = transitions[0];
        completion.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        list->ResourceBarrier(1, &completion);
        transitions[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
      }
      if (chosen.enhanced || before == D3D12_RESOURCE_STATE_COPY_SOURCE) list->ResourceBarrier(1, &transitions[1]);
      else list->ResourceBarrier(2, transitions);
      if (!preservation_only) set_attempt(value.provider, status::recorded);
      if (nominate_source && !preservation_only) record_frame_generation_copy(value);
      if (diagnostic) { diagnostic->result = status::recorded; diagnostic->stage = record_stage::recorded; }
      if (preserved) *preserved = {found->id, reinterpret_cast<std::uint64_t>(found->texture->resource.p),
        found->texture->identity, found->texture};
      return found->id;
    }
  }
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
  std::uint64_t record(std::uint64_t command, const input &value, record_diagnostic *diagnostic) {
    return record_impl(command, value, diagnostic, false);
  }
#endif
  static diagnostic_ticket record_auxiliary_texture(std::uint64_t command, const input &value,
      record_diagnostic *diagnostic, bool reusable_storage,
      texture_state_policy state_policy = texture_state_policy::source_contract) {
    try {
      const auto id = record_impl(command, value, diagnostic, true, nullptr, false, true, reusable_storage, state_policy);
      if (!id) return {};
      std::lock_guard lock(mutex);
      for (const auto &entry : diagnostic_slots) if (entry.id == id) return {id, entry.texture};
    } catch (...) {
      // Like depth preservation, every allocation precedes the injected copy.
      // An allocation failure never escapes into the application's API call.
      if (diagnostic) { diagnostic->result = status::exhausted; diagnostic->stage = record_stage::capacity; }
    }
    return {};
  }
  diagnostic_ticket record_diagnostic_texture(std::uint64_t command, const input &value, record_diagnostic *diagnostic,
      texture_state_policy state_policy) {
    return record_auxiliary_texture(command, value, diagnostic, false, state_policy);
  }
  diagnostic_ticket record_local_texture(std::uint64_t command, const input &value, record_diagnostic *diagnostic,
      texture_state_policy state_policy) {
    return record_auxiliary_texture(command, value, diagnostic, true, state_policy);
  }
  void finish_diagnostic_texture(const diagnostic_ticket &ticket, bool successful) {
    if (!ticket) return;
    std::lock_guard lock(mutex);
    for (auto &entry : diagnostic_slots) if (entry.id == ticket.id && entry.texture == ticket.ownership) {
      if (entry.finished) return;
      entry.finished = true; entry.success = successful;
      if (!successful) invalidate(entry, capture_failure::evaluation_failed);
      return;
    }
  }
  static status acquire_auxiliary_texture(const diagnostic_ticket &ticket, diagnostic_texture &out,
      std::uint64_t consumer_queue, bool local) {
    out = {};
    const auto result = [&](status value) { out.result = value; return value; };
    if (!ticket) return result(status::unavailable);
    if (local && !consumer_queue) return result(status::unsupported_queue);
    std::unique_lock lock(mutex);
    // A queue already registered (observe_present, once per Present) needs no
    // second interface check, observation and lock round trip here.
    if (local && !known_queue(consumer_queue)) {
      lock.unlock();
      observe_queue(consumer_queue);
      lock.lock();
    }
    const auto *consumer = local ? queue(consumer_queue) : nullptr;
    if (local && (!consumer || consumer->retiring)) return result(status::unsupported_queue);
    const auto native_consumer = consumer ? reinterpret_cast<std::uint64_t>(consumer->queue.p) : 0;
    for (auto &entry : diagnostic_slots) if (entry.id == ticket.id && entry.texture == ticket.ownership) {
      retire_dead_recordings(entry);
      out.capture_id = entry.id; out.producer_queue = entry.queue; out.producer_fence = entry.producer_fence;
      out.failure = entry.failure; out.producer_recording_retired = producer_recording_retired(entry);
      out.basis = entry.basis;
      const auto progress = local && entry.queue == native_consumer ? queue_progress{} : producer_progress(entry);
      out.producer_completed = progress.completed;
      const auto admission = local ? local_texture_admission(entry, native_consumer, consumer->device_identity, progress) :
        diagnostic_admission(entry, progress);
      if (admission != status::ready) return result(admission);
      const auto &texture = entry.texture;
      const auto desc = texture->resource->GetDesc();
      out.ownership = texture;
      out.texture = reinterpret_cast<std::uint64_t>(texture->resource.p);
      out.shared_handle = reinterpret_cast<std::uint64_t>(texture->shared_handle);
      out.device = reinterpret_cast<std::uint64_t>(texture->device);
      out.device_identity = texture->device_identity;
      out.resource_id = entry.metadata.source ? entry.metadata.source->cookie : 0;
      out.width = static_cast<std::uint32_t>(desc.Width); out.height = desc.Height; out.format = desc.Format;
      out.area = sunshine_api_contract::normalized_extent(entry.metadata.resource.area, out.width, out.height);
      return result(status::ready);
    }
    return result(status::unavailable);
  }
  status acquire_diagnostic_texture(const diagnostic_ticket &ticket, diagnostic_texture &out) {
    return acquire_auxiliary_texture(ticket, out, 0, false);
  }
  status acquire_local_texture(const diagnostic_ticket &ticket, std::uint64_t consumer_queue, diagnostic_texture &out) {
    return acquire_auxiliary_texture(ticket, out, consumer_queue, true);
  }
  void release_diagnostic_texture(const diagnostic_ticket &ticket) {
    if (!ticket) return;
    std::lock_guard lock(mutex);
    for (auto &entry : diagnostic_slots) if (entry.id == ticket.id && entry.texture == ticket.ownership) {
      entry.diagnostic_released = true;
      // A canceled in-progress API call cannot become an accepted diagnostic.
      // Its command/fence obligations survive regardless of ticket ownership.
      if (!entry.finished) { entry.finished = true; entry.success = false; }
      break;
    }
    collect_diagnostics();
  }
  std::uint64_t nominate(std::uint64_t command, const input &value, record_diagnostic *diagnostic) {
    // An API identifies the contents at this call, not whichever partial pass
    // Generic later preserves from the same resource. Keep both adapters on
    // the shared immutable snapshot/retirement path; never substitute a
    // clear-count heuristic for the middleware's capture boundary. An
    // undeclared state needs this recording's observation (copy_state::select).
    // The copy of an evaluation begun earlier (begin_evaluation: NGX, DLSS-G)
    // is its nomination in progress as nominate_evaluation's is.
    bool pending = false;
    if (requested.load() && valid_provider(value.provider)) {
      std::lock_guard lock(mutex);
      pending = mark_pending(evaluations[provider_index(value.provider)], value, false);
    }
    struct nomination_scope {
      const input &value;
      bool pending;
      ~nomination_scope() { if (pending) end_nomination(value); }
    } completion{value, pending};
    return record_impl(command, value, diagnostic, false, nullptr, true);
  }
  std::uint64_t evaluation_nomination::nominate(std::uint64_t command, const input &value, record_diagnostic *diagnostic) {
    // Its own copy was published at begin; another value (or none begun)
    // takes nominate's own path.
    if (nominated.pending && nominated.provider == value.provider &&
        same_nomination({nominated.epoch, nominated.sequence, 0, nominated.source_id, nominated.viewport}, nomination_of(value)))
      return record_impl(command, value, diagnostic, false, nullptr, true);
    return depth_capture::nominate(command, value, diagnostic);
  }
  std::uint64_t nominate_evaluation(std::uint64_t command, const input &value,
      std::uint64_t superseded_source_id, record_diagnostic *diagnostic) {
    return nominate_evaluation(command, &value, 1, superseded_source_id, diagnostic);
  }
  std::uint64_t nominate_evaluation(std::uint64_t command, const input *candidates, std::size_t count,
      std::uint64_t superseded_source_id, record_diagnostic *diagnostic) {
    const auto malformed = [&] {
      if (diagnostic) {
        *diagnostic = {}; diagnostic->command = command;
        diagnostic->result = status::malformed; diagnostic->stage = record_stage::malformed_input;
      }
      return std::uint64_t{};
    };
    if (!candidates || !count || count > 3) return malformed();
    const auto &value = candidates[0];
    for (std::size_t i = 1; i < count; ++i) {
      const auto &other = candidates[i];
      if (other.provider != value.provider || other.epoch != value.epoch || other.sequence != value.sequence ||
          other.source_id != value.source_id || other.viewport != value.viewport ||
          other.frame_generation_input != value.frame_generation_input) return malformed();
    }
    const bool pending = begin_nomination(value, superseded_source_id);
    struct nomination_scope {
      const input &value;
      bool pending;
      ~nomination_scope() { if (pending) end_nomination(value); }
    } completion{value, pending};
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
    if (pending) wait_nomination_gate();
#endif
    struct fallback_nomination {
      std::uint64_t ticket{};
      record_diagnostic diagnostic;
      ~fallback_nomination() { if (ticket) finish(ticket, false); }
      std::uint64_t release() { return std::exchange(ticket, 0); }
    } fallback;
    for (std::size_t i = 0; i < count; ++i) {
      record_diagnostic captured;
      // A candidate of the nomination published above needs no second
      // capture-lock round trip to find it published.
      const auto ticket = pending ? record_impl(command, candidates[i], &captured, false, nullptr, true) :
        nominate(command, candidates[i], &captured);
      if (diagnostic) *diagnostic = captured;
      if (!ticket) continue;
      if (captured.result == status::recorded) return ticket;
      if (!fallback.ticket) { fallback.ticket = ticket; fallback.diagnostic = captured; }
      else finish(ticket, false);
    }
    if (fallback.ticket && diagnostic) *diagnostic = fallback.diagnostic;
    return fallback.release();
  }
  preservation_ticket record_preserved(std::uint64_t command, std::uint64_t source,
      const sunshine_api_contract::contract &boundary, std::uint64_t queue, record_diagnostic *diagnostic) {
    try {
      input value;
      value.source = retain_source(source, diagnostic);
      if (!value.source) return {};
      // A depth-stencil source is tracked from its creation; one a full
      // table forgot is tracked again, unproven until its next barrier or
      // clear proves it.
      if (value.source->desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) track_deferred_source(value.source);
      value.resource.native = source;
      sunshine_api_contract::apply(boundary, value);
      value.source_present_generation = source_present_generation(value.source);
      // The same queue pointer the submission hook reports.
      com_ptr<ID3D12CommandQueue> checked_queue;
      queue = queue && query_native(queue, IID_ID3D12CommandQueue, checked_queue) ?
        reinterpret_cast<std::uint64_t>(checked_queue.p) : 0;
      preservation_ticket ticket;
      record_impl(command, value, diagnostic, true, &ticket, false, false, false, texture_state_policy::source_contract, queue);
      return ticket;
    } catch (...) {
      // Allocation failure must not escape a ReShade recording callback. All
      // allocations precede recording the copy; no partial ticket is published.
      if (diagnostic) {
        *diagnostic = {}; diagnostic->command = command; diagnostic->resource = source;
        diagnostic->result = status::exhausted; diagnostic->stage = record_stage::capacity;
      }
      return {};
    }
  }
  preservation_ticket record_preserved(std::uint64_t command, std::uint64_t source,
      std::uint32_t native_state, record_diagnostic *diagnostic) {
    return record_preserved(command, source, sunshine_api_contract::generic::preserved(native_state), 0, diagnostic);
  }
  void resource_created(std::uint64_t resource, std::uint32_t initial_state) {
    if (!requested.load() || !resource) return;
    com_ptr<ID3D12Resource> checked;
    if (!query_native(resource, IID_ID3D12Resource, checked)) return;
    // Buffers and simultaneous-access textures decay to COMMON when each
    // execution completes (MS Learn, Implicit State Transitions): no barrier
    // proves their state between executions.
    const auto desc = checked->GetDesc();
    if (desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER || (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS)) return;
    const auto identity = retain_source_cookie(checked.p);
    if (!identity) return;
    std::lock_guard lock(mutex);
    forget_destroyed();
    proven.track(identity, initial_state != UINT32_MAX, initial_state);
  }
  void resource_destroyed(std::uint64_t identity) {
    if (!identity || !requested.load()) return;
    for (auto &slot : destroyed) {
      std::uint64_t free{};
      if (slot.compare_exchange_strong(free, identity, std::memory_order_acq_rel)) {
        destroyed_pending.store(true, std::memory_order_release);
        return;
      }
    }
  }
  void track_deferred_source(const source_ref &source) {
    if (!requested.load() || !source || !source->cookie || source->desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER ||
        (source->desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS)) return;
    std::lock_guard lock(mutex);
    forget_destroyed();
    proven.adopt(source->cookie);
  }
  void observe_depth_clear(std::uint64_t native, std::uint64_t resource) {
    if (!requested.load() || !native || !resource) return;
    // No capture lock, exactly as barriers(): the recording thread owns it.
    auto owner = command(native, native_observer::get_recording_cookie(native), false);
    if (!owner || owner->closed || !owner->reported) return;
    // A tracked texture already has its identity.
    const auto identity = source_cookie(reinterpret_cast<ID3D12Object *>(resource));
    if (!identity) return;
    try {
      if (owner->clears.capacity() == 0) owner->clears.reserve(4);
      owner->clears.push_back({identity, ++owner->barrier_order});
    } catch (...) { invalidate(*owner, recording_loss::source_state_capacity); }
  }
  bool legacy_render_target_copy(std::uint64_t native, std::uint64_t resource) {
    com_ptr<ID3D12Resource> checked;
    if (!query_native(resource, IID_ID3D12Resource, checked) ||
        (checked->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS)) return false;
    if (!requested.load()) return true;
    const auto identity = retain_source_cookie(checked.p);
    if (!identity) return false;
    // No capture lock for the recording, exactly as barriers().
    com_ptr<ID3D12GraphicsCommandList> list;
    if (query_native(native, IID_ID3D12GraphicsCommandList, list)) {
      native = reinterpret_cast<std::uint64_t>(list.p);
      auto owner = native_observer::states_ready(native) ?
        command(native, native_observer::get_recording_cookie(native), false) : command_storage::handle{};
      if (owner && owner->reported && !owner->closed && !owner->invalid)
        if (const auto *entry = state(*owner, identity, false))
          return !entry->enhanced && entry->known && !entry->blocked && !entry->pending_split &&
            entry->value == D3D12_RESOURCE_STATE_RENDER_TARGET;
    }
    std::lock_guard lock(mutex);
    forget_destroyed();
    proven.watch(identity);
    return !proven.latched(identity).latched;
  }
  bool last_proven_state(std::uint64_t resource, std::uint64_t queue, proven_state &out) {
    out = {};
    com_ptr<ID3D12Resource> checked;
    com_ptr<ID3D12CommandQueue> checked_queue;
    if (!requested.load() || !query_native(resource, IID_ID3D12Resource, checked) ||
        !query_native(queue, IID_ID3D12CommandQueue, checked_queue)) return false;
    queue = reinterpret_cast<std::uint64_t>(checked_queue.p);
    const auto identity = source_cookie(checked.p);
    if (!identity || !native_observer::queue_ready(queue)) return false;
    std::lock_guard lock(mutex);
    out = proven.query(identity, queue, !enhanced_barriers_seen.load(std::memory_order_relaxed));
    return bool(out);
  }
  bool read_conflicted(std::uint64_t capture) { return native_observer::queue_order_conflicted(capture); }
  bool queue_order_line(std::uint64_t now, char *out, std::size_t size) {
    static std::uint64_t logged_at{}, logged_total{UINT64_MAX};
    static bool disabled_logged{};
    const auto s = native_observer::queue_order_counts();
    if (!s.enabled) {
      if (disabled_logged) return false;
      disabled_logged = true;
      std::snprintf(out, size, "Sunshine 3D queue order: disabled (QueueFenceOrder=0)");
      return true;
    }
    disabled_logged = false;
    const auto inflight = order_inflight_reads.load(std::memory_order_relaxed);
    std::uint64_t total = s.model.signals + s.model.waits + s.model.cpu_signals + s.ordered + s.resets + inflight +
      s.model.late_conflict + s.signal_hooks + s.wait_hooks + s.fence_signal_hooks + s.pending;
    for (const auto value : s.refused) total += value;
    for (const auto value : s.model.poisoned) total += value;
    if (total == logged_total || (logged_at && now - logged_at < 10000)) return false;
    logged_at = now; logged_total = total;
    using sunshine_queue_order::poison;
    using sunshine_queue_order::refusal;
    const auto p = [&](poison why) { return static_cast<unsigned long long>(s.model.poisoned[static_cast<unsigned>(why)]); };
    const auto r = [&](refusal why) { return static_cast<unsigned long long>(s.refused[static_cast<unsigned>(why)]); };
    std::snprintf(out, size,
      "Sunshine 3D queue order: hooks={signal=%u wait=%u fence_signal=%u pending=%u} queues=%llu fences=%llu settled=%llu "
      "signals=%llu waits=%llu cpu_signals=%llu poisoned={shared=%llu foreign=%llu cpu=%llu second=%llu monotonic=%llu "
      "unobserved=%llu removed=%llu unrecorded=%llu} refused={disabled=%llu pending_hooks=%llu uncovered=%llu untracked=%llu "
      "overlap=%llu unresolved=%llu baseline=%llu unsettled=%llu poisoned=%llu evicted=%llu cross_device=%llu} ordered=%llu "
      "proofs={inflight=%llu} evicted_waits=%llu resets=%llu cpu_unhooked=%llu late_conflict=%llu",
      s.signal_hooks, s.wait_hooks, s.fence_signal_hooks, s.pending, static_cast<unsigned long long>(s.model.queues),
      static_cast<unsigned long long>(s.model.fences), static_cast<unsigned long long>(s.model.settled),
      static_cast<unsigned long long>(s.model.signals), static_cast<unsigned long long>(s.model.waits),
      static_cast<unsigned long long>(s.model.cpu_signals), p(poison::shared), p(poison::foreign_device), p(poison::cpu_signal),
      p(poison::second_signaller), p(poison::non_monotonic), p(poison::unobserved_signal), p(poison::removed),
      p(poison::unrecorded_signaller), r(refusal::disabled), r(refusal::pending_hooks), r(refusal::uncovered),
      r(refusal::untracked), r(refusal::overlap), r(refusal::unresolved), r(refusal::baseline), r(refusal::unsettled),
      r(refusal::poisoned), r(refusal::evicted), r(refusal::cross_device), static_cast<unsigned long long>(s.ordered),
      static_cast<unsigned long long>(inflight), static_cast<unsigned long long>(s.model.evicted_waits),
      static_cast<unsigned long long>(s.resets), static_cast<unsigned long long>(s.cpu_unhooked),
      static_cast<unsigned long long>(s.model.late_conflict));
    return true;
  }
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
    return out;
  }
  void finish(std::uint64_t ticket, bool successful, capture_failure failure) {
    std::lock_guard lock(mutex);
    for (auto &value : slots) if (value.id == ticket) {
      value.finished = true; value.success = successful;
      if (!successful) invalidate(value, failure == capture_failure::none ? capture_failure::evaluation_failed : failure);
      // The view's run of rejected evaluations (expire_association) and its
      // newest accepted copy (frame_generation_scope_live).
      const auto &m = value.metadata;
      if (!value.preservation_only && valid_provider(m.provider))
        for (auto &view : evaluations[provider_index(m.provider)].views) {
          if (view.epoch != m.epoch || view.viewport != m.viewport || view.source_id != m.source_id) continue;
          // A compute-list copy is accepted once readable (accept_compute_copy).
          if (successful && !value.nomination_only && !value.invalid && !value.compute_list) {
            view.accepted_tick = std::max(view.accepted_tick, m.tick);
            view.accepted_sequence = std::max(view.accepted_sequence, m.sequence);
          }
          if (m.sequence < view.result_sequence) continue;
          view.result_sequence = m.sequence;
          if (successful) view.rejected_since = 0;
          else if (!view.rejected_since) view.rejected_since = GetTickCount64();
        }
      accept_compute_copy(value);
      return;
    }
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
          if (sunshine_freshness::fresh_lt(view.tick, selection_tick, sunshine_freshness::source_age_ms))
            ++diagnostic->active_view_count;
        }
      }
    }
    // A genuine source interruption revokes snapshots from before the gap.
    // Normal pending/repeated frames do not move this admission watermark.
    if (owner->provider_established &&
        chosen.result != status::ready && chosen.result != status::recorded && chosen.result != status::submitted)
      owner->admission_after = serial.load(std::memory_order_relaxed);
    if (diagnostic) diagnostic->newest_sequence = chosen.latest ? chosen.latest->metadata.sequence :
      chosen.pending_nomination ? chosen.pending_nomination->sequence : 0;
    if (diagnostic && chosen.pending_nomination) {
      const auto &pending = *chosen.pending_nomination;
      diagnostic->source_tick_ms = pending.tick;
    }
    if (!chosen.value) {
      owner->choice = describe_choice(chosen, decision, *owner, policy, present, selection_tick, false);
      owner->chosen = true; reader = owner->cookie;
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
    owner->chosen = true; reader = owner->cookie;
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
    if (!owner || owner->present != present || owner->present_capture != value.capture_id ||
        !owner->provider_established || owner->provider != value.metadata.provider ||
        owner->nominated_epoch != value.metadata.epoch) return;
    owner->last_epoch = value.metadata.epoch;
    owner->last_sequence = value.metadata.sequence;
    owner->last_source_tick = value.metadata.tick;
  }
  // An immediate consumer is a ReShade runtime's immediate list, open outside
  // any render pass during its present/effects events. The runtime submits it
  // on the consumer queue and resets it afterwards, never replaying it. Once
  // registered (observe_runtime_list) it has a lifecycle like any game list.
  // Unregistered, its read is released at that list's next observed
  // submission, followed by the queue fence.
  static bool consume_owned(std::uint64_t native, const packet &value, std::uint64_t destination,
      std::uint32_t destination_state, consumer_diagnostic *diagnostic, bool auxiliary = false, bool local_auxiliary = false,
      bool immediate = false, local_view *lease = nullptr, std::uint32_t lease_format = 0) {
    if (diagnostic) *diagnostic = {};
    const auto result = [diagnostic](consumer_status why) {
      if (diagnostic) diagnostic->result = why;
      return why == consumer_status::ready;
    };
    // A snapshot released with its device (release_device) has no resource.
    if (!native || !value.ownership || value.ownership->released.load(std::memory_order_acquire))
      return result(consumer_status::missing_input);
    const auto submitted_list = native;
    // ReShade's registered immediate list was checked for this exact interface
    // at registration and observed by this Present's observe_present: skip
    // the repeated interface query and observation (and their lock).
    const bool registered = immediate && runtime_list(native);
    com_ptr<ID3D12GraphicsCommandList> checked;
    ID3D12GraphicsCommandList *list = nullptr;
    auto cookie = registered ? native_observer::get_recording_cookie(native) : 0;
    if (registered && cookie) list = reinterpret_cast<ID3D12GraphicsCommandList *>(native);
    else {
      if (!query_native(native, IID_ID3D12GraphicsCommandList, checked)) return result(consumer_status::unsupported_interface);
      native = reinterpret_cast<std::uint64_t>(checked.p);
      observe_command(native);
      cookie = native_observer::get_recording_cookie(native);
      list = checked.p;
    }
    com_ptr<ID3D12Device> device;
    if (list->GetType() != D3D12_COMMAND_LIST_TYPE_DIRECT) return result(consumer_status::unsupported_command_type);
    if (FAILED(list->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void **>(device.put())))) return result(consumer_status::missing_device);
    const auto device_identity = device_cookie(device.p);
    if (diagnostic) diagnostic->device_identity = device_identity;
    if (!device_identity) return result(consumer_status::missing_device_identity);
    // The snapshot's own identity: its device may be gone (release_device).
    const auto expected_device_identity = value.ownership->device_identity;
    if (diagnostic) diagnostic->expected_device_identity = expected_device_identity;
    if (device_identity != expected_device_identity) return result(consumer_status::device_mismatch);
    com_ptr<ID3D12Resource> target;
    source_ref retained_target;
    if (destination) {
      if (!query_native(destination, IID_ID3D12Resource, target) || target.p == value.ownership->resource.p ||
          !copy_state::supported(destination_state)) return result(consumer_status::invalid_destination);
      com_ptr<ID3D12Device> target_device;
      if (FAILED(target->GetDevice(IID_ID3D12Device, reinterpret_cast<void **>(target_device.put()))) ||
          device_cookie(target_device.p) != device_identity) return result(consumer_status::device_mismatch);
      const auto source_desc = value.ownership->resource->GetDesc(), target_desc = target->GetDesc();
      if (target_desc.Width != source_desc.Width || target_desc.Height != source_desc.Height ||
          (auxiliary ? !diagnostic_description(target_desc) ||
            auxiliary_copy_format(target_desc.Format) != auxiliary_copy_format(source_desc.Format) :
            !supported_description(target_desc) || typeless(target_desc.Format) != typeless(source_desc.Format)))
        return result(consumer_status::invalid_destination);
      if (auxiliary) {
        retained_target = retain_source(reinterpret_cast<std::uint64_t>(target.p));
        if (!retained_target) return result(consumer_status::invalid_destination);
      }
    }
    std::unique_lock lock(mutex);
    // A list in ReShade's lifecycle (or a registered runtime list) is read
    // under its recording lease, which also proves it is open and outside a
    // render pass. An unregistered runtime immediate list relies on its
    // runtime's contract and a submission lease.
    auto recording = command(native, cookie, true);
    const bool by_submission = lifecycle_view(recording ? &*recording : nullptr) != list_view::open;
    if (by_submission && !immediate) return result(consumer_status::observer_not_ready);
    // A lease needs the recording lease of a registered runtime list.
    if (lease && (by_submission || !registered)) return result(consumer_status::observer_not_ready);
    if (!by_submission) {
      if (diagnostic) { diagnostic->cookie = cookie; diagnostic->invalidation = recording->invalidation; }
      if (recording->invalid) return result(consumer_status::recording_invalid);
    } else recording = {};
    bool matching_id = false;
    auto *begin = auxiliary ? diagnostic_slots.data() : slots.data();
    auto *end = begin + (auxiliary ? diagnostic_slots.size() : slots.size());
    for (auto *current = begin; current != end; ++current) if (current->id == value.capture_id) {
      auto &entry = *current;
      matching_id = true;
      // The copy-out, once made, is read without the recording's Reset.
      const bool reads_copy_out = !auxiliary && entry.copied_out && entry.published && entry.published == value.ownership;
      if (entry.texture != value.ownership && !reads_copy_out) continue;
      if ((entry.consumer_queue || !auxiliary) && entry.consumer_queue != value.queue)
        return result(consumer_status::ownership_mismatch);
      retire_dead_recordings(entry);
      if (diagnostic) diagnostic->slot_invalid = entry.invalid;
      if (entry.invalid || !entry.finished || !entry.success) return result(consumer_status::capture_not_ready);
      if (auxiliary) {
        const auto *consumer = known_queue(value.queue);
        if (!consumer || consumer->retiring || consumer->device_identity != device_identity)
          return result(consumer_status::device_mismatch);
        const auto progress = local_auxiliary && entry.queue == value.queue ? queue_progress{} : producer_progress(entry);
        const auto admission = local_auxiliary ? local_texture_admission(entry, value.queue, device_identity, progress) :
          diagnostic_admission(entry, progress);
        if (admission != status::ready)
          return result(consumer_status::capture_not_ready);
        if (entry.auxiliary_destination && entry.auxiliary_destination != retained_target)
          return result(consumer_status::invalid_destination);
        // A snapshot leased on this recording sits in the shader-resource
        // state; a copy here would assume COMMON.
        if (entry.lease_list == native && entry.lease_cookie == cookie && (target.p || !lease))
          return result(consumer_status::ownership_mismatch);
        if (lease) {
          // Direct binding is same-queue only (no foreign or mixed producer),
          // reads the storage's own view, and never follows a copy.
          if (entry.queue != value.queue || !entry.texture->views.p) return result(consumer_status::capture_not_ready);
          const auto storage = entry.texture->resource->GetDesc().Format;
          const auto requested = static_cast<DXGI_FORMAT>(lease_format);
          if (!same_color_family(storage, requested) ||
              (entry.lease_list == native && entry.lease_cookie == cookie && entry.texture->view_format != requested))
            return result(consumer_status::invalid_destination);
        }
      }
      // One unsubmitted immediate list per slot; repeated reads share it.
      if (by_submission && entry.pending_immediate && entry.pending_immediate != submitted_list)
        return result(consumer_status::ownership_mismatch);
      for (unsigned i = 0; i != entry.consumers.size(); ++i) if (by_submission || !entry.consumers[i] || entry.consumers[i] == cookie) {
        if (!entry.preservation_only && entry.queue != value.queue) {
          // Recheck the actual slot before recording any read. A pending copy
          // must never add a queue dependency to the application's schedule.
          // A copy still in flight is read only on the runtime's immediate
          // list, which its runtime executes on value.queue at this Present,
          // once the game's own fences order that queue after it (U-C1).
          const auto *consumer = known_queue(value.queue);
          const auto progress = producer_progress(entry);
          sunshine_queue_order::verdict relied;
          if (!consumer || consumer->retiring || !(reads_copy_out || producer_recording_retired(entry)) ||
              !readable_elsewhere(entry, immediate ? value.queue : 0, progress, &relied))
            return result(consumer_status::capture_not_ready);
          if (relied.ordered) {
            order_inflight_reads.fetch_add(1, std::memory_order_relaxed);
            native_observer::queue_order_rely(relied, entry.id);
          }
          if (diagnostic) {
            diagnostic->read_order = relied.ordered ? 3 : 2;
            diagnostic->producer_queue = entry.producer_position.queue;
            diagnostic->producer_ticket = entry.producer_position.ticket;
          }
        } else if (diagnostic) diagnostic->read_order = 1;
        if (by_submission) entry.pending_immediate = submitted_list;
        else { entry.consumers[i] = cookie; entry.consumer_recordings[i] = recording.life(); }
        entry.acquired = true;
        if (auxiliary) {
          entry.consumer_queue = value.queue;
          entry.auxiliary_destination = retained_target;
        }
        if (target.p) {
          // The registered consumer keeps the slot from reuse until its fence
          // passes, and the packet owns the storage: record without the lock.
          // All selected sources use this exact copy and consumer lease path.
          // Full subresource: depth plane or native auxiliary color pixels.
          ID3D12Resource *const source = value.ownership->resource.p;
          lock.unlock();
          native_observer::suppression_scope suppress;
          D3D12_RESOURCE_BARRIER transitions[2]{};
          transitions[0].Type = transitions[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
          transitions[0].Transition = {source, 0,
            auxiliary ? D3D12_RESOURCE_STATE_COMMON : value.ownership->rest, D3D12_RESOURCE_STATE_COPY_SOURCE};
          transitions[1].Transition = {target.p, 0, static_cast<D3D12_RESOURCE_STATES>(destination_state), D3D12_RESOURCE_STATE_COPY_DEST};
          list->ResourceBarrier(destination_state == D3D12_RESOURCE_STATE_COPY_DEST ? 1 : 2, transitions);
          D3D12_TEXTURE_COPY_LOCATION source_location{}, target_location{};
          source_location.pResource = source; source_location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
          target_location.pResource = target.p; target_location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
          list->CopyTextureRegion(&target_location, 0, 0, 0, &source_location, nullptr);
          std::swap(transitions[0].Transition.StateBefore, transitions[0].Transition.StateAfter);
          std::swap(transitions[1].Transition.StateBefore, transitions[1].Transition.StateAfter);
          list->ResourceBarrier(destination_state == D3D12_RESOURCE_STATE_COPY_DEST ? 1 : 2, transitions);
        } else if (lease) {
          auto &texture = *entry.texture;
          const auto requested = static_cast<DXGI_FORMAT>(lease_format);
          if (texture.view_format != requested || !texture.shader_resource) {
            const auto handle = texture.views->GetCPUDescriptorHandleForHeapStart();
            D3D12_SHADER_RESOURCE_VIEW_DESC view{};
            view.Format = requested; view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; view.Texture2D.MipLevels = 1;
            texture.device->CreateShaderResourceView(texture.resource.p, &view, handle);
            texture.shader_resource = handle.ptr;
            texture.view_format = requested;
          }
          if (entry.lease_list != native || entry.lease_cookie != cookie) {
            native_observer::suppression_scope suppress;
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition = {texture.resource.p, 0, D3D12_RESOURCE_STATE_COMMON, sampled_state};
            list->ResourceBarrier(1, &barrier);
            entry.lease_list = native; entry.lease_cookie = cookie;
          }
          const auto desc = texture.resource->GetDesc();
          *lease = {};
          lease->view = texture.shader_resource;
          lease->resource = reinterpret_cast<std::uint64_t>(texture.resource.p);
          lease->capture_id = entry.id;
          lease->width = static_cast<std::uint32_t>(desc.Width); lease->height = desc.Height;
          lease->format = desc.Format; lease->view_format = requested;
        }
        return result(consumer_status::ready);
      }
      invalidate(entry, capture_failure::consumer_capacity); reason = status::exhausted; return result(consumer_status::consumer_capacity);
    }
    return result(matching_id ? consumer_status::ownership_mismatch : consumer_status::slot_missing);
  }
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
  bool mark_consumer(std::uint64_t command, const packet &value, consumer_diagnostic *diagnostic) {
    if (!value.pixel_ready) {
      if (diagnostic) { *diagnostic = {}; diagnostic->result = consumer_status::capture_not_ready; }
      return false;
    }
    return consume_owned(command, value, 0, 0, diagnostic);
  }
#endif
  bool copy_current(std::uint64_t command, const packet &value, std::uint64_t destination,
      std::uint32_t destination_state, consumer_diagnostic *diagnostic, bool immediate) {
    if (!destination || !value.pixel_ready) {
      if (diagnostic) { *diagnostic = {}; diagnostic->result = !destination ? consumer_status::invalid_destination : consumer_status::capture_not_ready; }
      return false;
    }
    return consume_owned(command, value, destination, destination_state, diagnostic, false, false, immediate);
  }
  static bool copy_auxiliary_texture(std::uint64_t command, std::uint64_t consumer_queue,
      const diagnostic_ticket &ticket, std::uint64_t destination, std::uint32_t destination_state,
      consumer_diagnostic *diagnostic, bool local, bool immediate, local_view *lease = nullptr,
      std::uint32_t lease_format = 0) {
    const auto reject = [&](consumer_status value) {
      if (diagnostic) { *diagnostic = {}; diagnostic->result = value; }
      return false;
    };
    if (!command || !consumer_queue || !ticket) return reject(consumer_status::missing_input);
    if (!destination && !lease) return reject(consumer_status::invalid_destination);
    // A queue already registered (observe_present, once per Present) needs no
    // second interface check, observation and lock round trip here.
    packet selected;
    {
      std::unique_lock lock(mutex);
      if (!known_queue(consumer_queue)) {
        lock.unlock();
        observe_queue(consumer_queue);
        lock.lock();
      }
      auto *owner = queue(consumer_queue);
      if (!owner || owner->retiring) return reject(consumer_status::missing_device);
      const auto found = std::find_if(diagnostic_slots.begin(), diagnostic_slots.end(),
        [&](const auto &entry) { return entry.id == ticket.id; });
      if (found == diagnostic_slots.end()) return reject(consumer_status::slot_missing);
      if (found->texture != ticket.ownership) return reject(consumer_status::ownership_mismatch);
      if (found->texture->device_identity != owner->device_identity) return reject(consumer_status::device_mismatch);
      const auto native_consumer = reinterpret_cast<std::uint64_t>(owner->queue.p);
      const auto progress = local && found->queue == native_consumer ? queue_progress{} : producer_progress(*found);
      const auto admission = local ? local_texture_admission(*found, native_consumer, owner->device_identity, progress) :
        diagnostic_admission(*found, progress);
      if (admission != status::ready)
        return reject(consumer_status::capture_not_ready);
      selected.capture_id = found->id;
      selected.ownership = ticket.ownership;
      selected.device = reinterpret_cast<std::uint64_t>(found->texture->device);
      selected.queue = reinterpret_cast<std::uint64_t>(owner->queue.p);
    }
    return consume_owned(command, selected, destination, destination_state, diagnostic, true, local, immediate, lease,
      lease_format);
  }
  bool copy_diagnostic_texture(std::uint64_t command, std::uint64_t consumer_queue,
      const diagnostic_ticket &ticket, std::uint64_t destination, std::uint32_t destination_state,
      consumer_diagnostic *diagnostic, bool immediate) {
    return copy_auxiliary_texture(command, consumer_queue, ticket, destination, destination_state, diagnostic, false, immediate);
  }
  bool copy_local_texture(std::uint64_t command, std::uint64_t consumer_queue,
      const diagnostic_ticket &ticket, std::uint64_t destination, std::uint32_t destination_state,
      consumer_diagnostic *diagnostic, bool immediate) {
    return copy_auxiliary_texture(command, consumer_queue, ticket, destination, destination_state, diagnostic, true, immediate);
  }
  bool lease_local_view(std::uint64_t command, std::uint64_t consumer_queue, const diagnostic_ticket &ticket,
      std::uint32_t view_format, local_view &out, consumer_diagnostic *diagnostic) {
    out = {};
    if (!view_format) {
      if (diagnostic) { *diagnostic = {}; diagnostic->result = consumer_status::invalid_destination; }
      return false;
    }
    if (!runtime_list(command)) {
      if (diagnostic) { *diagnostic = {}; diagnostic->result = consumer_status::observer_not_ready; }
      return false;
    }
    local_view leased;
    if (!copy_auxiliary_texture(command, consumer_queue, ticket, 0, 0, diagnostic, true, true, &leased, view_format))
      return false;
    out = leased;
    return true;
  }
  void end_local_views(std::uint64_t command) {
    if (!command || !runtime_list(command)) return;
    const auto cookie = native_observer::get_recording_cookie(command);
    std::array<D3D12_RESOURCE_BARRIER, diagnostic_slot_limit> barriers{};
    UINT count = 0;
    std::lock_guard lock(mutex);
    for (auto &entry : diagnostic_slots) {
      if (!entry.lease_list || entry.lease_list != command) continue;
      // A submitted recording already ended: simultaneous-access storage
      // decayed to COMMON at its execution.
      if (entry.lease_cookie == cookie && cookie && entry.texture) {
        auto &barrier = barriers[count++];
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = {entry.texture->resource.p, 0, sampled_state, D3D12_RESOURCE_STATE_COMMON};
      }
      entry.lease_list = entry.lease_cookie = 0;
    }
    if (!count) return;
    native_observer::suppression_scope suppress;
    reinterpret_cast<ID3D12GraphicsCommandList *>(command)->ResourceBarrier(count, barriers.data());
  }
  SUNSHINE_CAPTURE_INTERNAL {
    consumer_status preserved_admission(const slot &value, const preservation_ticket &ticket,
        std::uint64_t consumer, std::uint64_t consumer_device, std::uint64_t recording) {
      if (!value.id || value.id != ticket.id || !value.preservation_only || value.texture != ticket.ownership)
        return consumer_status::ownership_mismatch;
      if (value.invalid || !value.finished || !value.success ||
          (!value.producer_submitted && (!recording || value.command != recording)))
        return consumer_status::capture_not_ready;
      if (!value.metadata.source || value.metadata.source->device_identity != consumer_device)
        return consumer_status::device_mismatch;
      if (value.consumer_queue && value.consumer_queue != consumer) return consumer_status::ownership_mismatch;
      return consumer_status::ready;
    }
  }
  bool copy_preserved(std::uint64_t command, std::uint64_t consumer_queue, const preservation_ticket &ticket,
      std::uint64_t destination, std::uint32_t destination_state, consumer_diagnostic *diagnostic) {
    const auto reject = [&](consumer_status value) {
      if (diagnostic) { *diagnostic = {}; diagnostic->result = value; }
      return false;
    };
    if (!command || !consumer_queue || !ticket) return reject(consumer_status::missing_input);
    // ReShade's registered immediate list and its known queue were observed
    // by this Present's observe_present.
    if (!runtime_list(command) || !native_observer::get_recording_cookie(command)) observe_command(command);
    const auto cookie = native_observer::get_recording_cookie(command);
    packet selected;
    {
      std::unique_lock lock(mutex);
      if (!known_queue(consumer_queue)) {
        lock.unlock();
        observe_queue(consumer_queue);
        lock.lock();
      }
      auto *owner = queue(consumer_queue);
      if (!owner || owner->retiring) return reject(consumer_status::missing_device);
      const auto native_queue = reinterpret_cast<std::uint64_t>(owner->queue.p);
      auto found = std::find_if(slots.begin(), slots.end(), [&](const auto &entry) { return entry.id == ticket.id; });
      if (found == slots.end()) return reject(consumer_status::slot_missing);
      const auto admission = preserved_admission(*found, ticket, native_queue, owner->device_identity, cookie);
      if (admission != consumer_status::ready) return reject(admission);
      found->consumer_queue = native_queue;
      describe_packet(*found, native_queue, selected);
    }
    return copy_current(command, selected, destination, destination_state, diagnostic);
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
