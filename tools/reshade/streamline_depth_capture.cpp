// SPDX-License-Identifier: GPL-3.0-only
#include "streamline_depth_capture.h"
#include "capture_state_policy.h"
#include "streamline_native_observer.h"
#include "native_command_storage.h"
#include "native_resource_identity.h"
#include "../../src/game3d_debug_formats.h"
#include <d3d12.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <optional>
#include <cstdio>
#include <initializer_list>
#include <mutex>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
#include "depth_cache_update.h" // Selection regressions apply the provider's display policy.
#endif

namespace sunshine_streamline::depth_capture {
  namespace {
    template<class T> struct com_ptr {
      T *p{};
      ~com_ptr() { reset(); }
      com_ptr() = default;
      com_ptr(const com_ptr &) = delete;
      com_ptr &operator=(const com_ptr &) = delete;
      T *operator->() const { return p; }
      void reset() { if (auto *owned = std::exchange(p, nullptr)) owned->Release(); }
      T **put() { reset(); return &p; }
    };
    template<class T> bool query_native(std::uint64_t native, REFIID iid, com_ptr<T> &out) {
      // Boundary inputs are live COM objects, but their incoming interface may
      // be only IUnknown/base. Never read larger vtables before exact QI.
      return native && SUCCEEDED(reinterpret_cast<IUnknown *>(native)->QueryInterface(iid,
        reinterpret_cast<void **>(out.put()))) && out.p;
    }
    // One lazily allocated pool serves eight rotating source members plus
    // several recordings in flight and metadata nominations. No second pool
    // or per-source synchronization implementation belongs to Generic depth.
    constexpr unsigned slot_limit = 32, source_limit = 64, queue_limit = 4;
    constexpr unsigned pixel_slot_limit = slot_limit - 4; // Source metadata must survive a full GPU pool.
    constexpr unsigned diagnostic_slot_limit = 32;
    constexpr std::uint64_t diagnostic_byte_limit = 256ull * 1024 * 1024;
    constexpr D3D12_RESOURCE_STATES sampled_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    using copy_state::write_states;
    constexpr GUID source_guid = sunshine_native_identity::resource_guid;
    constexpr GUID queue_guid{0x091eab97, 0xfcc2, 0x47fb, {0x97, 0x3b, 0x02, 0x7c, 0x69, 0xb0, 0xd5, 0x26}};
    std::atomic<std::uint64_t> serial{1};
    std::atomic<status> reason{status::inactive};
    constexpr unsigned provider_count = 2;
    using provider_kind = sunshine_scene_depth::provider_kind;
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
    DXGI_FORMAT typeless(DXGI_FORMAT value) {
      switch (value) {
      case DXGI_FORMAT_D32_FLOAT: case DXGI_FORMAT_R32_TYPELESS: case DXGI_FORMAT_R32_FLOAT: return DXGI_FORMAT_R32_TYPELESS;
      case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: case DXGI_FORMAT_R32G8X24_TYPELESS: case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS: return DXGI_FORMAT_R32G8X24_TYPELESS;
      case DXGI_FORMAT_D24_UNORM_S8_UINT: case DXGI_FORMAT_R24G8_TYPELESS: case DXGI_FORMAT_R24_UNORM_X8_TYPELESS: return DXGI_FORMAT_R24G8_TYPELESS;
      case DXGI_FORMAT_D16_UNORM: case DXGI_FORMAT_R16_TYPELESS: case DXGI_FORMAT_R16_UNORM: return DXGI_FORMAT_R16_TYPELESS;
      default: return DXGI_FORMAT_UNKNOWN;
      }
    }
    DXGI_FORMAT view_format(DXGI_FORMAT value) {
      switch (typeless(value)) {
      case DXGI_FORMAT_R32_TYPELESS: return DXGI_FORMAT_R32_FLOAT;
      case DXGI_FORMAT_R32G8X24_TYPELESS: return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
      case DXGI_FORMAT_R24G8_TYPELESS: return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
      case DXGI_FORMAT_R16_TYPELESS: return DXGI_FORMAT_R16_UNORM;
      default: return DXGI_FORMAT_UNKNOWN;
      }
    }
    struct copy_region {
      std::uint32_t width{}, height{};
      sunshine_scene_depth::extent area;
    };
    bool supported_description(const D3D12_RESOURCE_DESC &desc) {
      return desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && desc.SampleDesc.Count == 1 &&
        desc.DepthOrArraySize == 1 && desc.MipLevels == 1 && desc.Width && desc.Width <= 16384 &&
        desc.Height && desc.Height <= 16384 && typeless(desc.Format) != DXGI_FORMAT_UNKNOWN;
    }
    DXGI_FORMAT diagnostic_storage_format(DXGI_FORMAT format) {
      // Color/alpha allocations may be typeless while their UI views are UNORM.
      // Copy those bytes into a compatible typed snapshot so both the renderer
      // and diagnostic host receive an explicit format (the UI capture owner
      // checks the same mapping before it records a copy).
      return static_cast<DXGI_FORMAT>(auxiliary_snapshot_format(static_cast<std::uint32_t>(format)));
    }
    static_assert(auxiliary_snapshot_format(DXGI_FORMAT_R8G8B8A8_TYPELESS) == std::uint32_t(DXGI_FORMAT_R8G8B8A8_UNORM) &&
      auxiliary_snapshot_format(DXGI_FORMAT_B8G8R8A8_TYPELESS) == std::uint32_t(DXGI_FORMAT_B8G8R8A8_UNORM) &&
      auxiliary_snapshot_format(DXGI_FORMAT_R10G10B10A2_TYPELESS) == std::uint32_t(DXGI_FORMAT_R10G10B10A2_TYPELESS));
    unsigned diagnostic_pixel_bytes(DXGI_FORMAT format) {
      return game3d_debug::pixel_bytes(static_cast<unsigned>(diagnostic_storage_format(format)));
    }
    bool diagnostic_description(const D3D12_RESOURCE_DESC &desc) {
      return desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && desc.SampleDesc.Count == 1 &&
        desc.DepthOrArraySize == 1 && desc.MipLevels == 1 && desc.Width && desc.Width <= 16384 &&
        desc.Height && desc.Height <= 16384 && diagnostic_pixel_bytes(desc.Format) &&
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
    // A typed view format of a color storage family; only families whose
    // view reinterpretation is a plain cast. Anything else is incompatible.
    DXGI_FORMAT color_family(DXGI_FORMAT value) {
      switch (value) {
      case DXGI_FORMAT_R8G8B8A8_TYPELESS: case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        return DXGI_FORMAT_R8G8B8A8_TYPELESS;
      case DXGI_FORMAT_B8G8R8A8_TYPELESS: case DXGI_FORMAT_B8G8R8A8_UNORM: case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        return DXGI_FORMAT_B8G8R8A8_TYPELESS;
      case DXGI_FORMAT_B8G8R8X8_TYPELESS: case DXGI_FORMAT_B8G8R8X8_UNORM: case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
        return DXGI_FORMAT_B8G8R8X8_TYPELESS;
      case DXGI_FORMAT_R10G10B10A2_TYPELESS: case DXGI_FORMAT_R10G10B10A2_UNORM: return DXGI_FORMAT_R10G10B10A2_TYPELESS;
      case DXGI_FORMAT_R16G16B16A16_TYPELESS: case DXGI_FORMAT_R16G16B16A16_FLOAT: case DXGI_FORMAT_R16G16B16A16_UNORM:
        return DXGI_FORMAT_R16G16B16A16_TYPELESS;
      case DXGI_FORMAT_R32G32B32A32_TYPELESS: case DXGI_FORMAT_R32G32B32A32_FLOAT: return DXGI_FORMAT_R32G32B32A32_TYPELESS;
      case DXGI_FORMAT_R11G11B10_FLOAT: return DXGI_FORMAT_R11G11B10_FLOAT;
      case DXGI_FORMAT_R8_TYPELESS: case DXGI_FORMAT_R8_UNORM: return DXGI_FORMAT_R8_TYPELESS;
      case DXGI_FORMAT_A8_UNORM: return DXGI_FORMAT_A8_UNORM;
      case DXGI_FORMAT_R16_TYPELESS: case DXGI_FORMAT_R16_FLOAT: case DXGI_FORMAT_R16_UNORM: return DXGI_FORMAT_R16_TYPELESS;
      case DXGI_FORMAT_R32_TYPELESS: case DXGI_FORMAT_R32_FLOAT: return DXGI_FORMAT_R32_TYPELESS;
      default: return DXGI_FORMAT_UNKNOWN;
      }
    }
    bool typeless_color(DXGI_FORMAT value) {
      switch (value) {
      case DXGI_FORMAT_R8G8B8A8_TYPELESS: case DXGI_FORMAT_B8G8R8A8_TYPELESS: case DXGI_FORMAT_B8G8R8X8_TYPELESS:
      case DXGI_FORMAT_R10G10B10A2_TYPELESS: case DXGI_FORMAT_R16G16B16A16_TYPELESS: case DXGI_FORMAT_R32G32B32A32_TYPELESS:
      case DXGI_FORMAT_R8_TYPELESS: case DXGI_FORMAT_R16_TYPELESS: case DXGI_FORMAT_R32_TYPELESS: return true;
      default: return false;
      }
    }
    // A legal SRV format of the storage without relaxed format casting: the
    // storage's own typed format, or a typed member of a typeless storage's
    // family. (An sRGB storage read through UNORM is not; it keeps the copy.)
    bool same_color_family(DXGI_FORMAT storage, DXGI_FORMAT view) {
      if (typeless_color(view) || color_family(view) == DXGI_FORMAT_UNKNOWN) return false;
      return view == storage || (typeless_color(storage) && color_family(view) == color_family(storage));
    }
    status source_region(const D3D12_RESOURCE_DESC &desc,
        const sunshine_scene_depth::resource_description &resource, copy_region &out) {
      out = {};
      if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || !desc.Width || desc.Width > 16384 ||
          !desc.Height || desc.Height > 16384) return status::malformed;
      const auto area = resource.area;
      if ((area.width == 0) != (area.height == 0) || (!area.width && (area.left || area.top)) ||
          (area.width && (area.left > desc.Width || area.top > desc.Height ||
            area.width > desc.Width - area.left || area.height > desc.Height - area.top)) ||
          (resource.width && resource.width != desc.Width) || (resource.height && resource.height != desc.Height))
        return status::malformed;
      // Always retain the full allocation, including packed depth/stencil
      // plane zero. Statistics and the shader apply this same active rectangle;
      // no boxed depth/stencil GPU copy or conversion pass is introduced.
      out.width = static_cast<std::uint32_t>(desc.Width);
      out.height = desc.Height;
      out.area = area.width ? area : sunshine_scene_depth::extent{0, 0, out.width, out.height};
      return status::ready;
    }
    status capture_region(const D3D12_RESOURCE_DESC &desc,
        const sunshine_scene_depth::resource_description &resource, copy_region &out) {
      if (!supported_description(desc)) { out = {}; return status::unsupported_resource; }
      return source_region(desc, resource, out);
    }
    // enhanced: value is the legacy equivalent of an enhanced Barrier layout
    // (S3 shadow provenance; admission treats it like a legacy state).
    struct source_state { std::uint64_t source{}; std::uint32_t value{}; bool known{}, blocked{}, enhanced{}; };
    struct command_state {
      std::uint64_t cookie{};
      std::uint64_t observation_generation{};
      std::uint64_t native_barrier_calls{}, native_transition_count{}, last_barrier_command{};
      // The number of alias/split resources in a game recording is not bounded
      // by the number of depth sources we capture. Reuse this storage on Reset.
      std::vector<source_state> states;
      bool closed{}, invalid{}, render_pass{};
      // ReShade's lifecycle (or a registered runtime list's observed
      // submissions) drives this recording.
      bool reported{};
      recording_loss invalidation{recording_loss::none};
      void restart(std::uint64_t next_cookie, std::uint64_t generation) noexcept {
        states.clear();
        cookie = next_cookie; observation_generation = generation;
        native_barrier_calls = native_transition_count = last_barrier_command = 0;
        closed = invalid = render_pass = reported = false;
        invalidation = recording_loss::none;
      }
    };
    using command_storage = sunshine_native_command::storage<command_state>;
    using recording_ref = std::shared_ptr<sunshine_native_command::recording_lifetime>;
    // Advanced under mutex (invalidate_all); read without it by the barrier
    // hooks, which keep their recording's payload without the capture lock.
    std::atomic<std::uint64_t> command_generation{1};
    struct queue_state {
      com_ptr<ID3D12CommandQueue> queue;
      com_ptr<ID3D12Device> device;
      com_ptr<ID3D12Fence> fence;
      std::uint64_t cookie{}, next_fence{}, last_epoch{}, last_sequence{}, nominated_epoch{}, present{}, present_capture{};
      std::uint64_t admission_after{}, last_source_tick{};
      std::uint64_t device_identity{};
      bool retiring{}, provider_established{};
      provider_kind provider{provider_kind::streamline};
      std::uint64_t source_id{};
      std::uint32_t viewport{};
    };
  }

  struct source_reference {
    com_ptr<ID3D12Resource> resource;
    com_ptr<ID3D12Device> device;
    D3D12_RESOURCE_DESC desc{};
    std::uint64_t cookie{}, device_identity{};
    mutable std::atomic<std::uint64_t> present_generation{1};
  };
  struct texture_reference {
    com_ptr<ID3D12Resource> resource;
    com_ptr<ID3D12Device> device;
    com_ptr<ID3D12DescriptorHeap> views;
    std::uint64_t shader_resource{}, device_identity{}, identity{};
    HANDLE shared_handle{};
    std::uint64_t allocation_bytes{};
    // Local auxiliary snapshots: the format of the SRV at shader_resource,
    // created on the first lease (lease_local_view); protected by mutex.
    DXGI_FORMAT view_format{DXGI_FORMAT_UNKNOWN};
    ~texture_reference() { if (shared_handle) CloseHandle(shared_handle); }
  };
  namespace {
    struct fence_point { std::uint64_t queue{}, value{}; };
    struct slot {
      std::shared_ptr<texture_reference> texture;
      input metadata;
      // One auxiliary copy destination per snapshot. Keep the actual resource
      // alive through consumer Reset AND GPU completion, including legal replay.
      source_ref auxiliary_destination;
      std::uint64_t id{}, command{}, queue{}, producer_fence{}, consumer_queue{};
      // Values are comparable only within their own private queue timeline.
      // Reset may discard recording identities, never these GPU obligations.
      std::array<fence_point, queue_limit> retirements{};
      recording_ref producer_recording;
      std::array<std::uint64_t, 4> consumers{};
      std::array<recording_ref, 4> consumer_recordings;
      // A runtime immediate list that recorded a read and has not yet been
      // observed in a submission on consumer_queue. See consume_owned.
      std::uint64_t pending_immediate{};
      bool finished{}, success{}, acquired{}, invalid{}, producer_submitted{}, retirement_unknown{};
      bool preservation_only{};
      bool diagnostic_only{}, diagnostic_released{}, diagnostic_reusable{};
      bool source_nominated{}, nomination_only{}, nomination_invalid{};
      status pixel_failure{status::unavailable};
      capture_failure failure{capture_failure::none};
      // The pre-copy state's provenance of an auxiliary snapshot (diagnostic;
      // nothing here admits, retires or reuses a slot).
      state_basis basis{state_basis::unknown};
      // An empty slot reserved by a record attempt that allocates its storage
      // without the lock (record_impl). Expires after reservation_timeout_ms;
      // publication never trusts it. reserved_bytes counts toward the
      // diagnostic budget while live.
      std::uint64_t reservation{}, reservation_tick{}, reserved_bytes{};
      bool reserved_live{}; // The reservation is for a live snapshot (its budget).
      // When a colour/mask slot's capture retired and it kept only its cached
      // live allocation (collect_diagnostics). poll() releases old idle storage.
      std::uint64_t idle_tick{};
      // Direct binding (lease_local_view): the runtime list and recording on
      // which this snapshot sits in the shader-resource state until
      // end_local_views returns it to COMMON.
      std::uint64_t lease_list{}, lease_cookie{};
    };
    std::array<slot, slot_limit> slots;
    std::array<slot, diagnostic_slot_limit> diagnostic_slots;
    bool diagnostic_slots_active{}; // Protected by mutex; idle callbacks skip this pool.

    template<class Operation> void for_each_capture(Operation operation) {
      for (auto &value : slots) operation(value);
      if (diagnostic_slots_active) for (auto &value : diagnostic_slots) operation(value);
    }
    std::array<std::weak_ptr<const source_reference>, source_limit> sources;
    std::array<std::unique_ptr<queue_state>, queue_limit> queues;
    struct evaluation_head {
      std::uint64_t epoch{}, sequence{}, tick{}, source_id{};
      std::uint32_t viewport{};
    };
    struct evaluation_namespace {
      std::array<evaluation_head, 4> views{};
      std::uint64_t epoch{}, sequence{}, source_id{};
      evaluation_head pending_nomination;
    };
    std::array<evaluation_namespace, provider_count> evaluations;

    bool matching_evaluation(const evaluation_head &head, const input &value) {
      return head.epoch == value.epoch && head.sequence == value.sequence && head.source_id == value.source_id &&
        head.viewport == value.viewport;
    }
    void end_nomination(const input &value) {
      std::lock_guard lock(mutex);
      if (valid_provider(value.provider)) {
        auto &pending = evaluations[provider_index(value.provider)].pending_nomination;
        if (matching_evaluation(pending, value)) pending = {};
      }
    }

    void invalidate(slot &value, capture_failure why) {
      if (value.failure == capture_failure::none) value.failure = why;
      value.invalid = true;
      if (why != capture_failure::producer_signal_failed && why != capture_failure::consumer_signal_failed &&
          why != capture_failure::consumer_capacity && why != capture_failure::consumer_queue_changed)
        value.nomination_invalid = true;
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
    queue_state *known_queue(std::uint64_t native) {
      for (auto &value : queues) if (value && reinterpret_cast<std::uint64_t>(value->queue.p) == native) return value.get();
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
      for (auto &value : queues) if (value && value->cookie == cookie) return value.get();
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
    void invalidate_all() {
      std::lock_guard lock(mutex);
      for_each_capture([](auto &value) { invalidate(value, capture_failure::observer_loss); });
      ++command_generation;
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
      if (!owner || owner->closed || !owner->reported) return;
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
        // and keeps its own alias rule (depth_addon.cpp).
        if (value.Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION) {
          ++owner->native_transition_count;
          // A split can precede the first tag/cookie. Preserve that source's
          // rejection until Reset without poisoning unrelated captured depth.
          if (value.Flags != D3D12_RESOURCE_BARRIER_FLAG_NONE) { block_source(*owner, value.Transition.pResource); continue; }
          if (value.Transition.Subresource != 0 && value.Transition.Subresource != D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES) continue;
          // A resource can reach its first SDK nomination after all of its
          // transitions were recorded. Preserve that live object's identity
          // now, just as split barriers already do. The observed state
          // remains local to this exact recording; it is never a global hint.
          const auto identity = observed_source_cookie(value.Transition.pResource);
          if (!identity) { invalidate(*owner, recording_loss::source_identity_unavailable); continue; }
          auto *known = state(*owner, identity, true);
          if (!known) { invalidate(*owner, recording_loss::source_state_capacity); continue; }
          if (known->blocked) continue;
          known->known = value.Flags == D3D12_RESOURCE_BARRIER_FLAG_NONE && copy_state::supported(value.Transition.StateAfter);
          known->value = value.Transition.StateAfter;
          known->enhanced = false;
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
    // The legacy state equivalent to an enhanced barrier layout, which the
    // capture's own legacy copy barriers may then use. Queue-specific and video
    // layouts have no legacy equivalent.
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
    // texture's state to its new layout's legacy equivalent, exactly as a
    // legacy transition does; a layout without one blocks the texture for this
    // recording. Other subresources (a stencil plane) keep subresource 0's state.
    void enhanced_textures(std::uint64_t native, std::uint64_t cookie, unsigned count,
        const native_observer::enhanced_texture *textures) {
      // No capture lock, exactly as barriers().
      auto owner = command(native, cookie, true);
      if (!owner || owner->closed || !owner->reported) return;
      for (unsigned i = 0; i != count; ++i) {
        const auto &texture = textures[i];
        if (!texture.resource || !texture.first_subresource) continue;
        const auto legacy = legacy_state(texture.layout_after);
        if (!legacy) { block_source(*owner, texture.resource); continue; }
        const auto identity = observed_source_cookie(texture.resource);
        if (!identity) { invalidate(*owner, recording_loss::source_identity_unavailable); continue; }
        auto *known = state(*owner, identity, true);
        if (!known) { invalidate(*owner, recording_loss::source_state_capacity); continue; }
        if (known->blocked) continue;
        known->known = copy_state::supported(*legacy);
        known->value = *legacy;
        known->enhanced = true;
      }
    }
    // S3 shadow provenance of an admitted copy's pre-copy state: an entry on the
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
    template<std::size_t N> bool identify_submissions(std::array<slot, N> &captures, std::uint32_t count,
        const native_observer::command_identity *values, std::array<bool, N> &producers,
        std::array<bool, N> &consumers) {
      bool used = false;
      for (unsigned n = 0; n != count; ++n) {
        const auto cookie = values[n].recording_cookie;
        for (unsigned i = 0; i != captures.size(); ++i) {
          auto &value = captures[i];
          if (!value.id) continue;
          if (value.pending_immediate && value.pending_immediate == values[n].native_command) { consumers[i] = true; used = true; }
          // Zero is an unobserved recording, not a producer whose recording was
          // retired by Reset. It must not change capture state or create a fence.
          if (!cookie) continue;
          if (value.command == cookie) {
            if (value.producer_submitted) invalidate(value, capture_failure::replay); // No replayed old frame.
            producers[i] = true; used = true;
          }
          for (const auto consumer : value.consumers) if (consumer && consumer == cookie) { consumers[i] = true; used = true; }
        }
      }
      return used;
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
    struct queue_progress {
      std::uint64_t device_identity{}, completed{};
      bool queried{};
      bool valid() const { return queried && completed != UINT64_MAX; }
    };
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
      // injected copy. Only Reset/destruction makes that copy immutable for a
      // foreign queue; post-Execute replay detection would be too late.
      if (!producer_recording_retired(value)) return status::submitted;
      // The game may already order this producer after future consumer work.
      // Adding a reverse GPU Wait here can form a cycle, even after flushing
      // the consumer's immediate list. Poll completion without adding an edge
      // to the game's queue graph; pending source metadata remains admissible.
      return progress.completed >= value.producer_fence ? status::ready : status::submitted;
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
    struct capture_pick {
      slot *value{};
      const slot *latest{};
      const evaluation_head *pending_nomination{};
      status result{status::unavailable};
      selection_reason selection{selection_reason::not_attempted};
      std::uint64_t consumed_completed_capture{};
      queue_progress progress;
    };
    // The shared source age bounds nomination, completed-snapshot admission and
    // liveness alike. A zero tick is unavailable, never age zero.
    bool within_source_age(std::uint64_t tick, std::uint64_t now) {
      return tick && now >= tick && now - tick < sunshine_scene_depth::maximum_source_age_ms;
    }
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
        if (value.epoch == current.epoch && value.sequence && now >= value.tick && now - value.tick < sunshine_scene_depth::maximum_source_age_ms) ++active_views;
      if (active_views != 1) {
        out.result = active_views ? status::ambiguous : status::stale;
        out.selection = active_views ? selection_reason::ambiguous_views : selection_reason::inactive_views;
        return out;
      }
      const auto &pending = current.pending_nomination;
      if (pending.sequence && pending.epoch == current.epoch && pending.sequence == current.sequence &&
          pending.source_id == current.source_id && now >= pending.tick && now - pending.tick < sunshine_scene_depth::maximum_source_age_ms &&
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
      if (now < out.value->metadata.tick || now - out.value->metadata.tick >= sunshine_scene_depth::maximum_source_age_ms) {
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
        out.source_id = (1ull << 63) | policy.viewport;
      }
      return out;
    }
    // Keep one completed API snapshot while the CPU records its successor.
    // Completion and nomination advance independently for SR as well as FG.
    // Requiring the latest nomination to finish can starve a pipelined game.
    // Finished pixels stay valid through later metadata events (camera reset,
    // tag changes, observation loss); only source identity and age bound them.
    // consumer: the reading queue. A snapshot already submitted on it is
    // ordered before the read by queue order, exactly as a current capture is
    // (submission_status); another queue needs a retired recording and a
    // completed producer fence.
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
            !metadata.tick || now < metadata.tick || now - metadata.tick >= sunshine_scene_depth::maximum_source_age_ms) continue;
        if (!consumer || value.queue != consumer) {
          if (!producer_recording_retired(value)) continue;
          const auto progress = producer_progress(value);
          if (!progress.valid() || progress.completed < value.producer_fence) continue;
        }
        if (!best || metadata.sequence > best->metadata.sequence ||
            (metadata.sequence == best->metadata.sequence && value.id > best->id)) best = &value;
      }
      return best;
    }
    bool same_depth_layout(const input &a, const input &b) {
      if (!a.source || !b.source) return false;
      const auto &left = a.source->desc;
      const auto &right = b.source->desc;
      return left.Width == right.Width && left.Height == right.Height && left.Format == right.Format &&
        a.resource.kind == b.resource.kind && a.resource.area.left == b.resource.area.left &&
        a.resource.area.top == b.resource.area.top && a.resource.area.width == b.resource.area.width &&
        a.resource.area.height == b.resource.area.height;
    }
    selection_reason completed_fallback_reason(const queue_state &owner, std::uint64_t present,
        const slot &latest, const slot &completed) {
      if (completed.id <= owner.admission_after) return selection_reason::completed_before_gap;
      if (completed.metadata.sequence >= latest.metadata.sequence) return selection_reason::completed_not_older;
      if (!same_depth_layout(completed.metadata, latest.metadata)) return selection_reason::layout_changed;
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
                ((1ull << 63) | chosen.pending_nomination->viewport)))) {
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
        const auto matches = [&](const sunshine_scene_depth::frame &value) {
          return value.provider == provider_kind::streamline && value.frame_generation_input &&
            value.epoch == policy.epoch && value.viewport == policy.viewport &&
            value.source_id == ((1ull << 63) | policy.viewport);
        };
        if ((fg.value && !matches(fg.value->metadata)) || (fg.latest && !matches(fg.latest->metadata)) ||
            (fg.pending_nomination && (fg.pending_nomination->epoch != policy.epoch ||
              fg.pending_nomination->viewport != policy.viewport ||
              fg.pending_nomination->source_id != ((1ull << 63) | policy.viewport)))) {
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
    template<class Complete> bool gpu_retired(const slot &value, Complete complete) {
      if (value.retirement_unknown || !value.producer_fence || !complete(fence_point{value.queue, value.producer_fence})) return false;
      for (const auto &point : value.retirements) if (point.queue && (!point.value || !complete(point))) return false;
      return true;
    }
    bool needs_submission_fence(const std::array<bool, slot_limit> &producers,
        const std::array<bool, slot_limit> &consumers) {
      for (unsigned i = 0; i != slots.size(); ++i)
        if ((!slots[i].nomination_only && producers[i]) || consumers[i]) return true;
      return false;
    }
    void submitted(std::uint64_t native, std::uint32_t count, const native_observer::command_identity *values) {
      std::lock_guard lock(mutex);
      auto *owner = queue(native);
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
      const bool needs_fence = auxiliary_used || needs_submission_fence(producers, consumers);
      const auto fence = needs_fence ? ++owner->next_fence : 0;
      // This callback runs AFTER the actual native ExecuteCommandLists. Signal
      // therefore follows the copied/consumed pixels on this queue.
      const bool signaled = !needs_fence || SUCCEEDED(owner->queue->Signal(owner->fence.p, fence));
      for (unsigned i = 0; i != slots.size(); ++i) {
        auto &value = slots[i];
        if (producers[i]) producer_submitted(value, native, value.nomination_only ? 0 : fence,
          value.nomination_only || signaled);
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
      if (value.texture.use_count() > 1) return false;
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
    // The device is kept alive by the recording's retained source.
    record_stage allocate_storage(const storage_plan &plan, std::shared_ptr<texture_reference> &out) {
      out.reset();
      try {
        auto texture = std::make_shared<texture_reference>();
        plan.device->AddRef(); texture->device.p = plan.device;
        texture->device_identity = plan.device_identity;
        texture->allocation_bytes = plan.allocation_bytes;
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
      const auto desc = value.nomination_only ? value.metadata.source->desc : value.texture->resource->GetDesc();
      out.metadata = value.metadata; out.ownership = value.texture;
      out.resource_id = value.metadata.source ? value.metadata.source->cookie : 0;
      if (value.nomination_only) out.device = reinterpret_cast<std::uint64_t>(value.metadata.source->device.p);
      else {
        out.texture = reinterpret_cast<std::uint64_t>(value.texture->resource.p);
        out.shader_resource = value.texture->shader_resource;
        out.device = reinterpret_cast<std::uint64_t>(value.texture->device.p);
      }
      out.queue = consumer; out.producer_queue = value.queue; out.capture_id = value.id; out.producer_fence = value.producer_fence;
      out.width = static_cast<std::uint32_t>(desc.Width); out.height = desc.Height;
      out.area = value.metadata.resource.area.width ? value.metadata.resource.area :
        sunshine_scene_depth::extent{0, 0, out.width, out.height};
      out.format = typeless(desc.Format);
      out.srv_format = view_format(desc.Format);
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
        if (!retained) owner.reset();
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
    void observed_submission(std::uint64_t queue, std::uint32_t count, const native_observer::command_identity *values) {
      submitted(queue, count, values);
      // ReShade resets its own list right after submitting it: the submitted
      // recording ends and the next one begins.
      for (std::uint32_t i = 0; i != count; ++i)
        if (runtime_list(values[i].native_command))
          try { lifecycle_recording(values[i].native_command, list_event::reset); } catch (...) {}
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
    for (auto &entry : queues) if (!entry) {
      auto next = std::make_unique<queue_state>();
      auto *value = reinterpret_cast<ID3D12CommandQueue *>(native);
      if (value->GetDesc().Type != D3D12_COMMAND_LIST_TYPE_DIRECT ||
          FAILED(value->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void **>(next->device.put()))) ||
          FAILED(next->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), reinterpret_cast<void **>(next->fence.put())))) {
        reason = status::unsupported_queue; return;
      }
      next->device_identity = device_cookie(next->device.p, true);
      if (!next->device_identity) { reason = status::unsupported_queue; return; }
      value->AddRef(); next->queue.p = value;
      next->cookie = ++serial;
      if (FAILED(value->SetPrivateData(queue_guid, sizeof(next->cookie), &next->cookie))) { reason = status::unsupported_queue; return; }
      entry = std::move(next); return;
    }
    reason = status::exhausted;
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
        if (value.id && !head(value) && reclaimable(value)) {
          slot idle_slot;
          idle_slot.texture = std::move(value.texture);
          // A record attempt may be allocating for this slot without the lock.
          idle_slot.reservation = value.reservation; idle_slot.reservation_tick = value.reservation_tick;
          idle_slot.reserved_bytes = value.reserved_bytes; idle_slot.reserved_live = value.reserved_live;
          value = std::move(idle_slot);
        }
        if (value.id || !value.texture) continue;
        if (idle + value.texture->allocation_bytes > idle_pixel_byte_limit) discard.push_back(std::move(value.texture));
        else idle += value.texture->allocation_bytes;
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
  namespace {
    // An established source keeps its queue mono through gaps only while its
    // API still evaluates it. A game that stops calling the API without a
    // release (DLSS -> TAA, a cutscene without DLSS) loses the association
    // after this silence, and Generic takes over through its usual hysteresis.
    constexpr std::uint64_t association_timeout_ms = 1000;
    // Requires mutex.
    void expire_association(queue_state &owner, std::uint64_t now) {
      if (!owner.provider_established || !valid_provider(owner.provider)) return;
      const auto &current = evaluations[provider_index(owner.provider)];
      // The owner's own evaluations: its source and epoch, so the malformed-
      // input marker (epoch 0, source 0) never keeps a Streamline SR owner
      // (source 0) associated.
      const auto recent = [&](const evaluation_head &head) {
        return head.source_id == owner.source_id && head.epoch == owner.nominated_epoch && head.tick && now >= head.tick &&
          now - head.tick <= association_timeout_ms;
      };
      if (recent(current.pending_nomination) || std::any_of(current.views.begin(), current.views.end(), recent)) return;
      owner.provider_established = false;
      owner.source_id = owner.last_epoch = owner.last_sequence = owner.last_source_tick = owner.nominated_epoch = owner.present_capture = 0;
      // Snapshots from before the silence never re-establish it.
      owner.admission_after = std::max(owner.admission_after, serial.load(std::memory_order_relaxed));
    }
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
      return head.epoch && head.tick && now_ms >= head.tick && now_ms - head.tick <= window_ms;
    };
    for (const auto &space : evaluations) {
      if (recent(space.pending_nomination)) return true;
      for (const auto &view : space.views) if (recent(view)) return true;
    }
    return false;
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
    std::lock_guard lock(mutex);
    const auto existing = source_cookie(resource);
    for (const auto &weak : sources) if (auto value = weak.lock()) if (value->cookie == existing && existing)
      return result(value, status::ready, record_stage::retained);
    auto entry = std::find_if(sources.begin(), sources.end(), [](const auto &value) { return value.expired(); });
    if (entry == sources.end()) { reason = status::exhausted; return result({}, status::exhausted, record_stage::capacity); }
    auto value = std::make_shared<source_reference>();
    if (FAILED(resource->QueryInterface(__uuidof(ID3D12Resource), reinterpret_cast<void **>(value->resource.put()))))
      return result({}, status::unsupported_resource, record_stage::source_interface);
    if (FAILED(resource->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void **>(value->device.put()))))
      return result({}, status::unsupported_queue, record_stage::source_device);
    value->desc = resource->GetDesc();
    if (diagnostic) {
      diagnostic->width = static_cast<std::uint32_t>(value->desc.Width);
      diagnostic->height = value->desc.Height; diagnostic->format = value->desc.Format;
      diagnostic->flags = value->desc.Flags;
      diagnostic->dimension = value->desc.Dimension; diagnostic->mip_levels = value->desc.MipLevels;
      diagnostic->array_size = value->desc.DepthOrArraySize; diagnostic->samples = value->desc.SampleDesc.Count;
    }
    value->device_identity = device_cookie(value->device.p, true);
    if (!value->device_identity) { reason = status::unsupported_queue; return result({}, status::unsupported_queue, record_stage::device_identity); }
    // Retention proves object identity/lifetime, not capture capability. The
    // common opportunity recorder validates the actual copy description.
    value->cookie = retain_source_cookie(resource);
    if (!value->cookie) return result({}, status::unsupported_resource, record_stage::source_cookie);
    *entry = value;
    return result(value, status::ready, record_stage::retained);
  }

  std::uint64_t source_present_generation(const source_ref &source) {
    return source ? source->present_generation.load(std::memory_order_acquire) : 0;
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
      value = {epoch, sequence, now, source_id, viewport}; return;
    }
    auto oldest = std::min_element(current.views.begin(), current.views.end(), [](const auto &a, const auto &b) { return a.sequence < b.sequence; });
    *oldest = {epoch, sequence, now, source_id, viewport};
  }

  void begin_evaluation(std::uint64_t epoch, std::uint64_t sequence, std::uint32_t viewport,
      provider_kind provider, std::uint64_t source_id, std::uint64_t superseded_source_id) {
    std::lock_guard lock(mutex);
    begin_evaluation_locked(epoch, sequence, viewport, provider, source_id, superseded_source_id);
  }

  static bool begin_nomination(const input &value, std::uint64_t superseded_source_id) {
    std::lock_guard lock(mutex);
    begin_evaluation_locked(value.epoch, value.sequence, value.viewport, value.provider, value.source_id, superseded_source_id);
    if (!requested.load() || !valid_provider(value.provider)) return false;
    auto &current = evaluations[provider_index(value.provider)];
    if (current.epoch != value.epoch || current.sequence != value.sequence || current.source_id != value.source_id)
      return false;
    if (value.provider != provider_kind::streamline || !value.frame_generation_input || !value.epoch ||
        !value.sequence || !value.source_id || !value.source || !value.resource.native) return false;
    current.pending_nomination = {value.epoch, value.sequence, GetTickCount64(), value.source_id, value.viewport};
    return true;
  }

#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
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
  }
#endif
  static bool source_lifetime_current(const input &value, std::uint64_t current_generation) {
    // A copy recorded synchronously inside the authenticated tag call that
    // supplied the resource is current by construction, for OnlyValidNow and
    // UntilPresent alike: an independent FG Present of an earlier frame cannot
    // end this call's lifetime. With multi frame generation such Presents land
    // between reading the tag and recording its copy on a fraction of real
    // frames. This does not extend a tag to a later evaluation or authorize a
    // delayed preservation copy; those still require an unchanged generation.
    if (value.at_tag_call || (value.valid_until == sunshine_scene_depth::lifetime::at_call && value.force_snapshot))
      return true;
    return value.source_present_generation && value.source_present_generation == current_generation;
  }

  static std::uint64_t record_impl(std::uint64_t native, const input &value, record_diagnostic *diagnostic,
      bool preservation_only, preservation_ticket *preserved = nullptr, bool nominate_source = false,
      bool diagnostic_copy = false, bool reusable_storage = false,
      texture_state_policy state_policy = texture_state_policy::source_contract) {
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
      if (!preservation_only) set_attempt(value.provider, result);
      if (nominate_source && valid_nomination) {
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
    if (!value.source) return reject(status::malformed, record_stage::missing_source);
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
    auto *list = checked.p;
    const auto command_type = list->GetType();
    if (diagnostic) diagnostic->command_type = command_type;
    if (command_type != D3D12_COMMAND_LIST_TYPE_DIRECT) return reject(status::unsupported_queue, record_stage::command_type);
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
    std::shared_ptr<texture_reference> prepared;
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
        (diagnostic_description(desc) ? source_region(desc, value.resource, region) : status::unsupported_resource) :
        capture_region(desc, value.resource, region);
      if (region_status != status::ready) return reject(region_status, record_stage::resource_region);
      if (owner->invalid) return reject(status::unavailable, record_stage::recording_invalid);
      // Unobserved lists (refused or moved method tables) use the declared state.
      const auto *observed = states_observed ? state(*owner, value.source->cookie, false) : nullptr;
      if (diagnostic && observed) {
        diagnostic->observed = true; diagnostic->observed_state = observed->value; diagnostic->blocked = observed->blocked;
      }
      const auto chosen = copy_state::resolve(state_rule, value.native_state, observed ?
        copy_state::observation{true, observed->known, observed->blocked, observed->value} : copy_state::observation{}, desc.Flags);
      if (!chosen.admitted) return reject(chosen.result, chosen.stage);
      const auto before = chosen.state;
      const auto basis = copy_basis(observed, chosen);
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
      const auto texture_format = diagnostic_copy ? diagnostic_storage_format(desc.Format) : typeless(desc.Format);
      const auto compatible_storage = [&](const auto &texture) {
        if (!texture || texture->device_identity != value.source->device_identity) return false;
        if (diagnostic_copy && bool(texture->shared_handle) == reusable_storage) return false;
        const auto existing = texture->resource->GetDesc();
        return existing.Width == region.width && existing.Height == region.height && existing.Format == texture_format;
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
      if (found == pixel_end) return reject(status::exhausted, record_stage::capacity);
      auto texture = found->texture;
      const auto resting_state = diagnostic_copy ? D3D12_RESOURCE_STATE_COMMON : sampled_state;
      if (!compatible_storage(texture)) {
        if (prepared && compatible_storage(prepared)) {
          // Storage allocated by an earlier attempt. Its budget was reserved on
          // the slot it reserved; a different slot must admit it again.
          if (diagnostic_copy && found->reservation != reservation &&
              !reserve_diagnostic_bytes(diagnostic_slots, found, prepared->allocation_bytes, reusable_storage, reservation, &discarded))
            return reject(status::exhausted, record_stage::capacity);
          texture = std::move(prepared);
        } else {
          // Bounded: a second allocation means the shape changed meanwhile.
          if (attempt >= 2) return reject(status::exhausted, record_stage::capacity);
          storage_plan plan;
          plan.device = value.source->device.p;
          plan.device_identity = value.source->device_identity;
          auto &destination = plan.desc;
          destination = desc; destination.Format = texture_format; destination.Flags = D3D12_RESOURCE_FLAG_NONE;
          destination.Width = region.width; destination.Height = region.height;
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
            plan.view = view_format(desc.Format);
            // Sizes the idle-storage budget (release_idle_slots).
            plan.allocation_bytes = plan.device->GetResourceAllocationInfo(0, 1, &destination).SizeInBytes;
          }
          clear_reservation(reservation);
          reservation = ++reservation_serial;
          found->reservation = reservation; found->reservation_tick = GetTickCount64();
          found->reserved_bytes = plan.allocation_bytes; found->reserved_live = diagnostic_copy && reusable_storage;
          lock.unlock();
          discarded.clear();
          prepared.reset();
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
          if (auto *hook = testing::allocation_hook.exchange(nullptr)) hook(testing::allocation_context.load());
#endif
          const auto stage = allocate_storage(plan, prepared);
          lock.lock();
          if (stage != record_stage::recorded) return reject(status::failed, stage);
          release_on_exit.keep = true;
          continue;
        }
      }
      // Publication: this attempt passed every check above with the lock held.
      discarded.push_back(std::move(found->texture));
      *found = {};
      if (reservation) clear_reservation(std::exchange(reservation, 0));
      found->texture = std::move(texture); found->metadata = value; found->metadata.native_state = before;
      found->id = ++serial; found->command = cookie; found->producer_recording = owner.life();
      found->basis = basis;
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
      transitions[1].Transition = {found->texture->resource.p, 0, resting_state, D3D12_RESOURCE_STATE_COPY_DEST};
      if (before == D3D12_RESOURCE_STATE_COPY_SOURCE) list->ResourceBarrier(1, &transitions[1]);
      else list->ResourceBarrier(2, transitions);
      D3D12_TEXTURE_COPY_LOCATION source{}, destination{};
      source.pResource = value.source->resource.p; source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      destination.pResource = found->texture->resource.p; destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      // Subresource zero is mip zero/array zero/plane zero. capture_region rejects
      // multi-mip, arrays and MSAA. Packed stencil planes remain untouched.
      list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
      std::swap(transitions[0].Transition.StateBefore, transitions[0].Transition.StateAfter);
      std::swap(transitions[1].Transition.StateBefore, transitions[1].Transition.StateAfter);
      if (before != D3D12_RESOURCE_STATE_COPY_SOURCE && !(before & write_states)) {
        // The asynchronous copy/reuse regression exposes partial later writes
        // when restoring only a read state here. Complete the injected source
        // read at an explicit write-state boundary before restoring the game's
        // exact state. This adds ordering, never a write to the game resource.
        auto completion = transitions[0];
        completion.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        list->ResourceBarrier(1, &completion);
        transitions[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
      }
      if (before == D3D12_RESOURCE_STATE_COPY_SOURCE) list->ResourceBarrier(1, &transitions[1]);
      else list->ResourceBarrier(2, transitions);
      if (!preservation_only) set_attempt(value.provider, status::recorded);
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
      out.device = reinterpret_cast<std::uint64_t>(texture->device.p);
      out.device_identity = texture->device_identity;
      out.resource_id = entry.metadata.source ? entry.metadata.source->cookie : 0;
      out.width = static_cast<std::uint32_t>(desc.Width); out.height = desc.Height; out.format = desc.Format;
      out.area = entry.metadata.resource.area.width ? entry.metadata.resource.area :
        sunshine_scene_depth::extent{0, 0, out.width, out.height};
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
    auto source = value;
    // An API identifies the contents at this call, not whichever partial pass
    // Generic later preserves from the same resource. Keep both adapters on
    // the shared immutable snapshot/retirement path; never substitute a
    // clear-count heuristic for the middleware's capture boundary.
    if (source.proof == sunshine_scene_depth::state_proof::unavailable)
      source.proof = sunshine_scene_depth::state_proof::observed_nonzero;
    return record_impl(command, source, diagnostic, false, nullptr, true);
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
      const auto ticket = nominate(command, candidates[i], &captured);
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
      std::uint32_t native_state, record_diagnostic *diagnostic) {
    try {
      input value;
      value.source = retain_source(source, diagnostic);
      if (!value.source) return {};
      value.resource.native = source;
      value.native_state = native_state;
      value.proof = sunshine_scene_depth::state_proof::declared;
      value.valid_until = sunshine_scene_depth::lifetime::until_present;
      value.source_present_generation = source_present_generation(value.source);
      preservation_ticket ticket;
      record_impl(command, value, diagnostic, true, &ticket);
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
  void finish(std::uint64_t ticket, bool successful, capture_failure failure) {
    std::lock_guard lock(mutex);
    for (auto &value : slots) if (value.id == ticket) {
      value.finished = true; value.success = successful;
      if (!successful) invalidate(value, failure == capture_failure::none ? capture_failure::evaluation_failed : failure);
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
      decision.source_id = (1ull << 63) | policy.viewport;
    }
    if (diagnostic) {
      *diagnostic = {}; diagnostic->requested_queue = native;
    }
    queue_progress progress;
    bool recording_retired = false;
    const auto result = [diagnostic, &decision, &progress, &recording_retired](status why, const slot *value = nullptr) {
      if (diagnostic) {
        diagnostic->result = why;
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
          if (selection_tick >= view.tick && selection_tick - view.tick < sunshine_scene_depth::maximum_source_age_ms)
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
    if (!chosen.value) return result(chosen.result, chosen.latest);
    auto *best = chosen.value;
    // Borrowed CPU ownership alone does not imply a GPU consumer. Only a
    // successful pre-read mark_consumer establishes that retirement obligation.
    bind_pixel_consumer(*best, native, chosen.result);
    establish_provider(*owner, *best, present);
    describe_packet(*best, native, out);
    out.pixel_ready = out.pixel_ready && chosen.result == status::ready;
    result(chosen.result, best);
    return true; // Source authority and current pixels are separate outcomes.
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
    if (!native || !value.ownership) return result(consumer_status::missing_input);
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
    const auto expected_device_identity = device_cookie(reinterpret_cast<ID3D12Device *>(value.device));
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
      if (entry.texture != value.ownership) continue;
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
          const auto *consumer = known_queue(value.queue);
          const auto progress = producer_progress(entry);
          if (!consumer || consumer->retiring || !producer_recording_retired(entry) ||
              !progress.valid() || !entry.producer_fence || progress.completed < entry.producer_fence)
            return result(consumer_status::capture_not_ready);
        }
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
            auxiliary ? D3D12_RESOURCE_STATE_COMMON : sampled_state, D3D12_RESOURCE_STATE_COPY_SOURCE};
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
      selected.device = reinterpret_cast<std::uint64_t>(found->texture->device.p);
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
  namespace {
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
    CASE(completed_already_consumed); CASE(presentation_already_selected); CASE(layout_changed);
    CASE(completed_not_readable); CASE(completed_snapshot); CASE(fg_scope_missing); CASE(fg_scope_mismatch);
#undef CASE
    }
    return "unknown";
  }
  const char *name(state_basis value) {
    switch (value) {
#define CASE(x) case state_basis::x: return #x
    CASE(unknown); CASE(observed_legacy); CASE(observed_enhanced); CASE(declared); CASE(contract);
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
#ifdef SUNSHINE_STREAMLINE_PROBE_TEST
  namespace testing {
    void age_idle_storage(std::uint64_t ms) {
      std::lock_guard lock(mutex);
      for (auto &value : diagnostic_slots)
        if (!value.id && value.texture) value.idle_tick -= std::min(value.idle_tick, ms);
    }
    bool diagnostic_snapshot_regression() {
      // Same admission helper as production; completion alone must not publish
      // a texture while its recorded copy can still legally replay.
      slot value;
      value.id = 1; value.diagnostic_only = value.preservation_only = true;
      value.command = 3; value.producer_recording = std::make_shared<sunshine_native_command::recording_lifetime>();
      value.producer_recording->cookie = 3;
      queue_progress complete{11, 9, true};
      if (diagnostic_admission(value, complete) != status::recorded) return false;
      value.finished = value.success = true;
      producer_submitted(value, 5, 9, true);
      if (diagnostic_admission(value, complete) != status::submitted) return false;
      retire_recording(value, 3);
      if (diagnostic_admission(value, {11, 8, true}) != status::submitted ||
          diagnostic_admission(value, complete) != status::ready) return false;
      if (diagnostic_admission(value, {11, UINT64_MAX, true}) != status::failed ||
          diagnostic_admission(value, {}) != status::failed) return false;
      value.diagnostic_released = true;
      if (diagnostic_admission(value, complete) != status::stale) return false;
      value.diagnostic_released = false;
      producer_submitted(value, 5, 10, false);
      if (diagnostic_admission(value, complete) != status::failed || reclaimable(value)) return false;
      // Rejection/cancellation cannot turn unknown retirement into a free slot.
      value.diagnostic_released = true;
      if (reclaimable(value)) return false;
      D3D12_RESOURCE_DESC desc{};
      desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; desc.Width = 64; desc.Height = 48;
      desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
      for (const auto format : {DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8_UINT, DXGI_FORMAT_R16G16B16A16_FLOAT,
          DXGI_FORMAT_R8G8B8A8_TYPELESS, DXGI_FORMAT_B8G8R8A8_TYPELESS}) {
        desc.Format = format;
        if (!diagnostic_description(desc)) return false;
      }
      for (const auto format : {DXGI_FORMAT_R8_TYPELESS, DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R16G16B16A16_TYPELESS,
          DXGI_FORMAT_B8G8R8X8_TYPELESS, DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_NV12, DXGI_FORMAT_BC1_UNORM}) {
        desc.Format = format;
        if (diagnostic_description(desc)) return false;
      }
      desc.Format = DXGI_FORMAT_R8_UNORM; desc.SampleDesc.Count = 2;
      if (diagnostic_description(desc)) return false;
      // Cached allocations count toward the live budget with live tickets, and
      // only the cache can be evicted to admit a changed output layout.
      constexpr auto limit = diagnostic_byte_limit;
      std::array<slot, diagnostic_slot_limit> cache;
      for (unsigned i = 0; i != 5; ++i) {
        cache[i].texture = std::make_shared<texture_reference>();
        cache[i].texture->allocation_bytes = limit / 4;
      }
      cache[0].id = 1; cache[0].diagnostic_reusable = true; // A live ticket survives budget pressure.
      // A small request keeps the fixed limit: 5/4 allocated, room for 1/16.
      if (!reserve_diagnostic_bytes(cache, &cache[9], limit / 16, true) ||
          !cache[0].texture || cache[1].texture || cache[2].texture || !cache[3].texture || !cache[4].texture) return false;
      // A large live request is sized from itself: room for eight of it.
      if (!reserve_diagnostic_bytes(cache, &cache[9], limit, true) || !cache[3].texture || !cache[4].texture) return false;
      // Dump copies have their own fixed budget: live storage never counts
      // against them and is never evicted for them.
      cache[5].texture = std::make_shared<texture_reference>();
      cache[5].texture->allocation_bytes = limit / 2;
      cache[5].id = 2; // A dump ticket.
      if (!reserve_diagnostic_bytes(cache, &cache[9], limit / 2, false) ||
          reserve_diagnostic_bytes(cache, &cache[9], limit / 2 + 1, false) ||
          !cache[3].texture || !cache[4].texture) return false;
      return !reserve_diagnostic_bytes(cache, &cache[0], 0, false) &&
        !reserve_diagnostic_bytes(cache, &cache[5], limit + 1, false) &&
        reserve_diagnostic_bytes(cache, &cache[5], limit, false);
    }
    bool record_diagnostic_regression() {
      // COM Release can synchronously invoke observers. Detach ownership first
      // so reentrant reset/put cannot release the same reference twice.
      struct reentrant_release {
        com_ptr<reentrant_release> *owner{};
        unsigned calls{};
        bool saw_empty{};
        void Release() {
          ++calls;
          saw_empty = !owner->p;
          if (calls == 1) owner->put();
        }
      };
      static_assert(!std::is_copy_constructible_v<com_ptr<reentrant_release>>);
      static_assert(!std::is_copy_assignable_v<com_ptr<reentrant_release>>);
      reentrant_release object;
      {
        com_ptr<reentrant_release> owner;
        object.owner = &owner;
        owner.p = &object;
        if (*owner.put() || object.calls != 1 || !object.saw_empty) return false;
        object.calls = 0;
        owner.p = &object;
      }
      if (object.calls != 1 || !object.saw_empty) return false;
      // Exercise real early rejection paths; no fake graphics vtable or GPU is
      // needed. A later caller can overwrite the shared status, not this result.
      initialize(true);
      input value;
      // An FG presentation can advance the device's Present generation while
      // the render thread is inside slSetTag. That cannot expire OnlyValidNow's
      // synchronous copy, while an ordinary retained tag still expires.
      value.source_present_generation = 7;
      for (const auto lifetime : {sunshine_scene_depth::lifetime::until_present,
          sunshine_scene_depth::lifetime::until_evaluation}) {
        value.valid_until = lifetime;
        if (!source_lifetime_current(value, 7) || source_lifetime_current(value, 8)) return false;
      }
      value.valid_until = sunshine_scene_depth::lifetime::at_call;
      value.force_snapshot = true;
      if (!source_lifetime_current(value, 8)) return false;
      value.force_snapshot = false;
      if (source_lifetime_current(value, 8)) return false;
      // The same holds for an UntilPresent tag copied inside its own tag call
      // (FG input), but not for that tag copied later at an evaluation.
      value.valid_until = sunshine_scene_depth::lifetime::until_present;
      value.at_tag_call = true;
      if (!source_lifetime_current(value, 8)) return false;
      value.at_tag_call = false;
      if (source_lifetime_current(value, 8)) return false;
      value = {};
      value.epoch = value.sequence = 1;
      value.resource.native = 7;
      record_diagnostic missing;
      missing.recording_closed = true; missing.width = 123;
      if (record(5, value, &missing) || missing.result != status::malformed ||
          missing.stage != record_stage::missing_source || missing.command != 5 || missing.resource != 7 ||
          missing.recording_closed || missing.width) return false;
      value.source = std::make_shared<source_reference>();
      record_diagnostic lifetime;
      if (record(5, value, &lifetime) || lifetime.result != status::unsupported_lifetime ||
          lifetime.stage != record_stage::unsupported_lifetime || last_status() != status::unsupported_lifetime ||
          missing.result != status::malformed || missing.stage != record_stage::missing_source) return false;
      // The optional observer does not change the rejection or shared status.
      if (record(5, value) || last_status() != lifetime.result) return false;
      value.valid_until = sunshine_scene_depth::lifetime::until_evaluation;
      record_diagnostic proof;
      if (record(5, value, &proof) || proof.result != status::unsupported_state ||
          proof.stage != record_stage::unsupported_proof) return false;
      shutdown();
      record_diagnostic inactive;
      if (record(5, value, &inactive) || inactive.result != status::malformed || inactive.stage != record_stage::inactive) return false;
      if (retain_source(7, &inactive) || inactive.result != status::inactive || inactive.stage != record_stage::inactive ||
          inactive.resource != 7 || inactive.command) return false;
      return true;
    }
    bool crop_region_regression() {
      D3D12_RESOURCE_DESC desc{};
      desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
      desc.Width = 2228; desc.Height = 1256;
      desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
      desc.Format = DXGI_FORMAT_R32_FLOAT;
      sunshine_scene_depth::resource_description resource;
      copy_region region;
      if (capture_region(desc, resource, region) != status::ready || region.area.left || region.area.top ||
          region.area.width != 2228 || region.area.height != 1256 || region.width != 2228 || region.height != 1256) return false;
      resource.area = {0, 0, 2227, 1252};
      if (capture_region(desc, resource, region) != status::ready || region.width != 2228 || region.height != 1256 ||
          region.area.left || region.area.top || region.area.width != 2227 || region.area.height != 1252) return false;
      resource.area = {1, 2, 2227, 1252};
      if (capture_region(desc, resource, region) != status::ready || region.area.left != 1 ||
          region.area.top != 2 || region.area.width != 2227 || region.area.height != 1252 ||
          region.width != 2228 || region.height != 1256) return false;
      resource.area.left = UINT32_MAX;
      if (capture_region(desc, resource, region) != status::malformed) return false;
      resource.area = {0, 0, 2228, 0};
      if (capture_region(desc, resource, region) != status::malformed) return false;
      resource.area = {1, 0, 0, 0};
      if (capture_region(desc, resource, region) != status::malformed) return false;
      resource.area = {}; resource.width = 2227;
      if (capture_region(desc, resource, region) != status::malformed) return false;
      resource.width = 0;
      const auto ordinary = desc;
      for (const auto format : {DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
          DXGI_FORMAT_R32G8X24_TYPELESS, DXGI_FORMAT_D24_UNORM_S8_UINT, DXGI_FORMAT_D16_UNORM}) {
        desc = ordinary; desc.Format = format;
        resource.area = {};
        if (capture_region(desc, resource, region) != status::ready ||
            region.width != 2228 || region.height != 1256 || region.area.width != 2228 || region.area.height != 1256) return false;
        resource.area = {1, 2, 2227, 1252};
        if (capture_region(desc, resource, region) != status::ready ||
            region.width != 2228 || region.height != 1256 || region.area.left != 1 || region.area.top != 2 ||
            region.area.width != 2227 || region.area.height != 1252) return false;
      }
      desc = ordinary; desc.Format = DXGI_FORMAT_R32_TYPELESS;
      if (capture_region(desc, resource, region) != status::ready) return false;
      desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
      if (capture_region(desc, resource, region) != status::ready || region.width != 2228 || region.height != 1256 ||
          region.area.left != 1 || region.area.top != 2 || region.area.width != 2227 || region.area.height != 1252) return false;
      resource.area = {};
      if (capture_region(desc, resource, region) != status::ready) return false;
      desc = ordinary; desc.MipLevels = 2;
      if (capture_region(desc, resource, region) != status::unsupported_resource) return false;
      desc = ordinary; desc.DepthOrArraySize = 2;
      if (capture_region(desc, resource, region) != status::unsupported_resource) return false;
      desc = ordinary; desc.SampleDesc.Count = 2;
      if (capture_region(desc, resource, region) != status::unsupported_resource) return false;
      sunshine_scene_depth::projection_basis projection;
      projection.reversed = true;
      projection.depth_offset = projection.depth_scale = NAN;
      if (!valid_projection(projection)) return false; // Opaque absent coefficients are never consumed.
      projection.supplied = true;
      if (valid_projection(projection)) return false;
      projection = {0, .0625, true, true};
      if (!valid_projection(projection)) return false;
      projection.reversed = false;
      if (valid_projection(projection)) return false;
      projection.depth_scale = -.0625;
      if (!valid_projection(projection)) return false;
      projection = {};
      projection.encoding = sunshine_scene_depth::depth_encoding::linear_distance;
      projection.direction_supplied = true;
      if (!valid_projection(projection)) return false; // Linear distance needs no camera matrix.
      projection.reversed = true;
      if (valid_projection(projection)) return false;
      projection.raw_scale = -2;
      if (!valid_projection(projection)) return false;
      projection.raw_scale = 0;
      if (valid_projection(projection)) return false;
      projection.raw_scale = -2; projection.raw_bias = NAN;
      if (valid_projection(projection)) return false;
      projection.raw_bias = 0; projection.direction_supplied = false;
      return !valid_projection(projection);
    }
    bool provider_admission_regression() {
      // Exercise the production acquisition policy without COM/GPU operations.
      // Pixel ownership/fences remain covered by the native runtime fixtures.
      struct cleanup {
        std::array<slot, slot_limit> saved_slots{slots};
        std::array<evaluation_namespace, provider_count> saved_evaluations{evaluations};
        std::array<status, provider_count> saved_attempts{attempt_status[0].load(), attempt_status[1].load()};
        bool saved_requested{requested.load()};
        status saved_reason{reason.load()};
        std::uint64_t saved_serial{serial.load()};
        ~cleanup() {
          slots = saved_slots; evaluations = saved_evaluations;
          for (unsigned i = 0; i != provider_count; ++i) attempt_status[i] = saved_attempts[i];
          requested = saved_requested; reason = saved_reason; serial = saved_serial;
        }
      } restore;
      slots = {}; evaluations = {}; requested = true;
      {
        // Handoff evidence (evaluation_live): any real evaluation, Streamline
        // SR's source 0 included (FG Off in a Streamline-only game holds the
        // queue in mono while SR's first capture is pending); never the
        // malformed-input marker, which alone uses epoch 0.
        begin_evaluation(5, 1, 0, provider_kind::streamline);
        if (!evaluation_live(GetTickCount64(), 250)) return false;
        evaluations = {};
        begin_evaluation(0, 2, UINT32_MAX, provider_kind::streamline);
        if (evaluation_live(GetTickCount64(), 250)) return false;
        // An established Streamline SR owner (source 0) stays associated
        // through its own epoch's evaluations only, never through the marker.
        queue_state sr;
        sr.provider_established = true; sr.provider = provider_kind::streamline; sr.nominated_epoch = 5;
        expire_association(sr, GetTickCount64());
        if (sr.provider_established) return false;
        evaluations = {};
        begin_evaluation(5, 3, 0, provider_kind::streamline);
        sr.provider_established = true; sr.nominated_epoch = 5;
        expire_association(sr, GetTickCount64());
        if (!sr.provider_established) return false;
        evaluations = {};
      }
      queue_state owner;
      constexpr std::uint64_t native = 77;
      {
        // These prerequisites belong to capture ownership, not the provider's
        // diagnostic consumer. Exercise each refusal without COM/GPU work.
        slot next;
        next.id = 301; next.metadata.provider = provider_kind::ngx;
        next.metadata.epoch = 4; next.metadata.sequence = 201;
        next.metadata.source_id = 7; next.metadata.viewport = 2;
        next.finished = next.success = next.producer_submitted = true;
        next.producer_fence = 12;
        capture_pick picked;
        picked.value = &next; picked.latest = &next;
        picked.result = status::submitted;
        picked.selection = selection_reason::completed_already_consumed;
        const auto classified = [&](const capture_pick &choice) { return classify_acquisition(choice, owner, {}); };
        const auto proof = classified(picked);
        if (!proof.source_selected || !proof.source_valid ||
            proof.provider != next.metadata.provider || proof.capture_id != next.id ||
            proof.epoch != next.metadata.epoch || proof.sequence != next.metadata.sequence ||
            proof.source_id != next.metadata.source_id || proof.viewport != next.metadata.viewport) return false;
        auto no_selection = picked; no_selection.value = nullptr;
        if (classified(no_selection).source_selected) return false;
        owner.last_epoch = next.metadata.epoch; owner.last_sequence = next.metadata.sequence;
        if (!classified(no_selection).repeated_frame) return false;
        no_selection.result = status::failed;
        if (classified(no_selection).repeated_frame) return false;
        next.finished = next.success = false;
        if (!classified(picked).source_valid) return false; // Unfinished is not a failed result.
        next.finished = true;
        if (classified(picked).source_valid) return false;
        capture_pick absent;
        const auto missing = classify_acquisition(absent, owner, {true, 4, 2});
        if (missing.source_selected || missing.source_valid || missing.pending_frame || missing.repeated_frame ||
            missing.epoch != 4 || missing.viewport != 2 || missing.source_id != ((1ull << 63) | 2)) return false;
        owner.last_epoch = owner.last_sequence = 0;
      }
      {
        // The continuity consumer needs a precise refusal reason, not merely
        // "submitted". Exercise the same pre-GPU fallback checks used below.
        auto source = std::make_shared<source_reference>();
        source->desc.Width = 160; source->desc.Height = 90; source->desc.Format = DXGI_FORMAT_R32_FLOAT;
        slot completed, latest;
        completed.id = 100; completed.metadata.epoch = 2; completed.metadata.sequence = 10;
        completed.metadata.source = source;
        completed.metadata.resource.area = {0, 0, 160, 90};
        latest = completed; latest.id = 101; latest.metadata.sequence = 11;
        queue_state consumed;
        consumed.last_epoch = 2; consumed.last_sequence = 10;
        consumed.present = 21; consumed.present_capture = latest.id;
        if (completed_fallback_reason(consumed, 21, latest, completed) != selection_reason::completed_already_consumed)
          return false;
        // Repeated effects in the same presentation must keep this evidence;
        // an existing pending admission is not a different-source failure.
        consumed.present = 20;
        if (completed_fallback_reason(consumed, 21, latest, completed) != selection_reason::completed_already_consumed)
          return false;
        ++latest.metadata.resource.area.top;
        if (completed_fallback_reason(consumed, 21, latest, completed) != selection_reason::layout_changed) return false;
        latest.metadata.resource.area.top = 0;
        consumed.admission_after = completed.id;
        if (completed_fallback_reason(consumed, 21, latest, completed) != selection_reason::completed_before_gap) return false;
        consumed.admission_after = 0; consumed.last_sequence = 9;
        if (completed_fallback_reason(consumed, 21, latest, completed) != selection_reason::completed_snapshot) return false;
        completed.metadata.sequence = latest.metadata.sequence;
        if (completed_fallback_reason(consumed, 21, latest, completed) != selection_reason::completed_not_older) return false;
      }
      const auto make = [&](unsigned index, provider_kind provider,
          std::uint64_t epoch, std::uint64_t sequence, std::uint64_t source) -> slot & {
        auto &value = slots[index]; value = {};
        value.id = ++serial; value.queue = native; value.producer_fence = 1;
        value.finished = value.success = value.producer_submitted = true;
        value.metadata.provider = provider; value.metadata.source_id = source;
        value.metadata.epoch = epoch; value.metadata.sequence = sequence;
        value.metadata.tick = GetTickCount64();
        begin_evaluation(epoch, sequence, 0, provider, source);
        return value;
      };
      const auto select = [&](std::uint64_t present) { return select_provider(owner, native, present, GetTickCount64()); };
      // An observed API with no usable capture cannot suppress another source.
      begin_evaluation(1, 100, 0, provider_kind::streamline);
      if (owner.provider_established || select(1).value ||
          select(1).selection != selection_reason::no_current_nomination) return false;
      auto &ngx = make(0, provider_kind::ngx, 1, 2, 99);
      ngx.finished = false;
      if (select(1).value || owner.provider_established) return false;
      ngx.finished = true; ngx.success = false;
      if (select(1).value) return false;
      ngx.success = true; ngx.producer_submitted = false;
      if (select(1).value) return false;
      ngx.producer_submitted = true; ngx.queue = native + 1;
      if (select(1).value) return false;
      ngx.queue = native;
      auto chosen = select(1);
      if (chosen.value != &ngx) return false;
      establish_provider(owner, ngx, 1);
      if (!owner.provider_established || owner.provider != provider_kind::ngx || owner.source_id != 99) return false;
      // Acquisition without a successful copy must not consume this frame.
      if (select(2).value != &ngx) return false;
      // complete_frame() bookkeeping for a successful copy.
      owner.last_epoch = ngx.metadata.epoch; owner.last_sequence = ngx.metadata.sequence;
      owner.last_source_tick = ngx.metadata.tick;
      // NGX's already copied frame is not a pending successor, so a ready SL
      // source takes priority on the next presentation; repeated effects
      // within the already-selected presentation remain on NGX.
      make(1, provider_kind::streamline, 1, 101, 0);
      if (select(1).value != &ngx || select(2).value != &slots[1]) return false;
      slots[1].metadata.frame_generation_input = true;
      slots[1].metadata.source_id = 1ull << 63;
      slots[1].metadata.sequence = 102;
      begin_evaluation(1, 102, 0, provider_kind::streamline, slots[1].metadata.source_id, 0);
      if (select(2).value != &slots[1] || select_provider(owner, native, 2, GetTickCount64(), {true, 1, 0}).value != &slots[1]) return false;
      queue_state takeover;
      takeover.provider_established = true; takeover.provider = provider_kind::ngx;
      takeover.source_id = 99; takeover.nominated_epoch = takeover.last_epoch = 1; takeover.last_sequence = 999;
      establish_provider(takeover, slots[1], 2);
      if (select_provider(takeover, native, 3, GetTickCount64()).value != &slots[1]) return false;
      slots[1].metadata.frame_generation_input = false;
      begin_evaluation(0, 9999, UINT32_MAX, provider_kind::streamline);
      if (select(1).value != &ngx) return false;
      begin_evaluation(1, 3, 0, provider_kind::ngx, 99);
      if (select(2).value || !owner.provider_established || owner.source_id != 99) return false;
      auto &resumed = make(2, provider_kind::ngx, 1, 3, 99);
      if (select(2).value != &resumed) return false;
      establish_provider(owner, resumed, 2);
      // Epoch restart may reset sequence; it does not inherit the old watermark.
      auto &recreated = make(3, provider_kind::ngx, 2, 1, 100);
      if (select(3).value != &recreated) return false;
      establish_provider(owner, recreated, 3);
      if (owner.source_id != 100 || owner.provider != provider_kind::ngx) return false;
      // Fresh queues prefer ready SL even when NGX captured first.
      slots = {}; evaluations = {};
      owner.provider_established = false; owner.present_capture = owner.last_sequence = owner.last_epoch = 0;
      make(0, provider_kind::ngx, 3, 1, 123);
      auto &preferred = make(1, provider_kind::streamline, 3, 1, 0);
      if (select(4).value != &preferred) return false;
      // Simultaneous distinct view/source identities remain ambiguous.
      begin_evaluation(3, 2, 1, provider_kind::ngx, 124);
      const auto ambiguous = select_capture(owner, native, 4, provider_kind::ngx, GetTickCount64());
      if (ambiguous.value || ambiguous.result != status::ambiguous ||
          ambiguous.selection != selection_reason::ambiguous_views) return false;
      {
        slots = {}; evaluations = {};
        queue_state preferred_owner;
        auto &fallback = make(0, provider_kind::ngx, 60, 1, 500);
        auto &preferred = make(1, provider_kind::streamline, 70, 1, 0);
        preferred.finished = false;
        const auto pick = [&](std::uint64_t presentation, selection_policy policy = {}) {
          return select_provider(preferred_owner, native, presentation, GetTickCount64(), policy);
        };
        if (pick(1).value != &fallback) return false;
        establish_provider(preferred_owner, fallback, 1);
        preferred_owner.last_epoch = 60; preferred_owner.last_sequence = 1;
        preferred_owner.last_source_tick = fallback.metadata.tick;
        // A submitted nomination without readable pixels must not take over.
        preferred.finished = true; preferred.nomination_only = true;
        preferred.pixel_failure = status::unsupported_resource;
        auto source = std::make_shared<source_reference>(); source->device_identity = 17;
        preferred.metadata.source = source; preferred.source_nominated = true;
        preferred_owner.device_identity = 17;
        if (pick(2).value == &preferred) return false;
        preferred.nomination_only = false; preferred.source_nominated = false;
        // The owner's already copied frame is not a pending successor; it does
        // not hold a readable capture from the other provider.
        if (pick(2).value != &preferred) return false;
        // A live owner that is still delivering keeps its source while its real
        // successor is pending: the other provider does not make selection alternate.
        make(2, provider_kind::ngx, 60, 2, 500).finished = false;
        if (pick(2).value == &preferred) return false;
        // An owner without a recently copied frame holds nothing.
        preferred_owner.last_source_tick = GetTickCount64() - sunshine_scene_depth::maximum_source_age_ms - 1;
        if (pick(2).value != &preferred) return false;
        // Metadata observed after the copy (camera reset, tag change, loss)
        // never makes a readable SL copy yield to a ready NGX source.
        preferred.metadata.observation_revision = 5;
        make(2, provider_kind::ngx, 60, 2, 500);
        queue_state unowned; unowned.device_identity = preferred_owner.device_identity;
        if (select_provider(unowned, native, 2, GetTickCount64()).value != &preferred) return false;
        establish_provider(preferred_owner, preferred, 2);
        preferred_owner.last_epoch = 70; preferred_owner.last_sequence = 1;
        preferred_owner.last_source_tick = preferred.metadata.tick;
        auto &next_fallback = make(2, provider_kind::ngx, 60, 2, 500);
        auto &pending = make(3, provider_kind::streamline, 70, 2, 0); pending.finished = false;
        auto held = pick(3);
        if (held.value || held.latest != &pending || held.result != status::recorded) return false;
        preferred_owner.last_source_tick = GetTickCount64() - sunshine_scene_depth::maximum_source_age_ms - 1;
        if (pick(3).value != &next_fallback) return false;
        preferred_owner.last_source_tick = GetTickCount64();
        pending.finished = true; pending.success = false;
        if (pick(3).value != &next_fallback) return false;
        pending.success = true;
        if (pick(3).value != &pending) return false;
        begin_evaluation(70, 3, 0, provider_kind::streamline);
        if (pick(3).value != &next_fallback) return false;
      }
      const auto empty_queue = std::find_if(queues.begin(), queues.end(), [](const auto &entry) { return !entry; });
      if (empty_queue == queues.end()) return false;
      *empty_queue = std::make_unique<queue_state>();
      struct release_queue { std::unique_ptr<queue_state> &entry; ~release_queue() { entry.reset(); } } queue_cleanup{*empty_queue};
      auto &established = **empty_queue;
      {
        // An observed FG mode is source authority even before SL has usable
        // pixels. Older NGX, ordinary SL SR, and a different viewport/epoch
        // must not sneak through that gap or supply its camera separately.
        slots = {}; evaluations = {};
        constexpr std::uint64_t fg_epoch = 42, fg_id = 1ull << 63;
        const selection_policy fg_policy{true, fg_epoch, 0};
        const auto pick = [&](std::uint64_t present, selection_policy policy = {}) {
          return select_provider(established, native, present, GetTickCount64(), policy);
        };
        auto &first_ngx = make(0, provider_kind::ngx, 30, 1, 400);
        if (pick(20).value != &first_ngx || pick(20, fg_policy).value) return false;
        establish_provider(established, first_ngx, 20);
        // Turning FG off again before its first admitted pixels must leave
        // the established ordinary source usable; intent does not own it.
        if (pick(21, fg_policy).value || pick(21).value != &first_ngx) return false;
        auto &sr = make(1, provider_kind::streamline, fg_epoch, 1, 0);
        sr.metadata.projection = {1, -1, true, false};
        if (pick(21, fg_policy).value) return false;
        auto &fg = make(2, provider_kind::streamline, fg_epoch, 2, fg_id);
        fg.metadata.frame_generation_input = true;
        fg.metadata.projection = {0, 1, true, true};
        fg.metadata.sequence = 3;
        begin_evaluation(fg_epoch, 3, 0, provider_kind::streamline, fg_id, 0);
        fg.finished = false;
        auto pending = pick(21, fg_policy);
        if (pending.value || pending.latest != &fg || pending.result != status::recorded ||
            established.provider != provider_kind::ngx) return false;
        fg.finished = true;
        const auto accepted = pick(21, fg_policy);
        if (accepted.value != &fg || accepted.value->metadata.projection.depth_offset != 0 ||
            accepted.value->metadata.projection.depth_scale != 1 ||
            pick(21, {true, fg_epoch + 1, 0}).value || pick(21, {true, fg_epoch, 1}).value) return false;
        // No prior provider: the same policy still beats the earlier NGX ID.
        queue_state fresh;
        if (select_provider(fresh, native, 21, GetTickCount64(), fg_policy).value != &fg) return false;
        queue_state sr_owner;
        establish_provider(sr_owner, sr, 20);
        sr_owner.last_epoch = fg_epoch; sr_owner.last_sequence = 1;
        if (select_provider(sr_owner, native, 21, GetTickCount64(), fg_policy).value != &fg) return false;
        // A transient missing camera does not invent one from NGX or revoke
        // otherwise valid SL depth; consumers retain the raw-depth path.
        fg.metadata.projection.supplied = false;
        if (pick(21, fg_policy).value != &fg || pick(21, fg_policy).value->metadata.projection.supplied) return false;
        fg.metadata.projection.supplied = true;
        establish_provider(established, fg, 21);
        established.last_epoch = fg_epoch; established.last_sequence = 3;
        begin_evaluation(fg_epoch, 4, 0, provider_kind::streamline, fg_id);
        make(3, provider_kind::ngx, 30, 2, 400);
        if (pick(22, fg_policy).value) return false;
        // A successful FG Off retires that exact source; subsequent normal
        // acquisition resumes a new NGX frame, never reviving pre-gap pixels.
        retire_source(provider_kind::streamline, fg_epoch, fg_id);
        if (established.provider_established || pick(23).value || pick(23, fg_policy).value) return false;
        auto &resumed_ngx = make(4, provider_kind::ngx, 30, 3, 400);
        if (pick(24).value != &resumed_ngx || pick(24, fg_policy).value) return false;
      }
      {
        // A failed/ambiguous FG option revokes FG authority without starving
        // independent ordinary depth on subsequent presentations.
        slots = {}; evaluations = {};
        queue_state revoked_owner;
        constexpr std::uint64_t fg_epoch = 42, fg_id = 1ull << 63;
        selection_policy excluded;
        excluded.exclude_unconfirmed_fg = true;
        const auto pick = [&](std::uint64_t present, selection_policy policy) {
          return select_provider(revoked_owner, native, present, GetTickCount64(), policy);
        };
        auto &old_fg = make(0, provider_kind::streamline, fg_epoch, 1, fg_id);
        old_fg.metadata.frame_generation_input = true;
        establish_provider(revoked_owner, old_fg, 30);
        revoked_owner.last_epoch = fg_epoch; revoked_owner.last_sequence = 1;
        revoked_owner.last_source_tick = old_fg.metadata.tick;
        make(1, provider_kind::ngx, 60, 1, 500);
        if (pick(30, excluded).value) return false; // Keep the existing presentation pin.
        revoked_owner.admission_after = serial.load(); // acquire's one-time gap watermark.
        auto &fallback = make(2, provider_kind::ngx, 60, 2, 500);
        fallback.metadata.projection.encoding = sunshine_scene_depth::depth_encoding::linear_distance;
        fallback.metadata.projection.raw_scale = 2;
        for (const auto present : {31ull, 32ull}) {
          const auto selected = pick(present, excluded);
          if (selected.value != &fallback || selected.result != status::ready ||
              selected.value->metadata.projection.encoding != sunshine_scene_depth::depth_encoding::linear_distance ||
              selected.value->metadata.projection.raw_scale != 2) return false;
        }
        fallback.success = false;
        const auto unavailable = pick(31, excluded);
        // The live failed NGX attempt is reported in front of the excluded FG
        // scope. It supplies no source and never reveals the excluded FG input.
        if (unavailable.value || unavailable.pending_nomination || unavailable.latest != &fallback ||
            unavailable.result != status::failed) return false;
        fallback.success = true;
        auto &pending_fg = make(3, provider_kind::streamline, fg_epoch, 2, fg_id);
        pending_fg.metadata.frame_generation_input = true; pending_fg.finished = false;
        if (pick(31, excluded).value != &fallback || pick(31, {true, fg_epoch, 0}).value == &fallback) return false;
        // Retirement invalidates old pixels and removes that FG view, without
        // suppressing a later independent ordinary SL source on this viewport.
        retire_source(provider_kind::streamline, fg_epoch, fg_id);
        if (!old_fg.invalid || !pending_fg.invalid || pick(32, {true, fg_epoch, 0}).value) return false;
        auto &ordinary_sl = make(4, provider_kind::streamline, fg_epoch, 3, 0);
        if (pick(31, excluded).value != &ordinary_sl) return false;
      }
      {
        // A failed options call revokes all captures in its exact FG scope,
        // including tags observed while the SDK call was in flight. A racing
        // newer capture may be discarded once; only a fresh tag restores FG.
        slots = {}; evaluations = {};
        constexpr std::uint64_t fg_epoch = 80, fg_id = 1ull << 63;
        auto &old_fg = make(0, provider_kind::streamline, fg_epoch, 10, fg_id);
        old_fg.metadata.frame_generation_input = true;
        auto &new_fg = make(1, provider_kind::streamline, fg_epoch, 12, fg_id);
        new_fg.metadata.frame_generation_input = true;
        establish_provider(established, new_fg, 100);
        established.last_epoch = fg_epoch; established.last_sequence = 12;
        retire_source(provider_kind::streamline, fg_epoch, fg_id);
        finish(new_fg.id, true);
        if (!old_fg.invalid || !new_fg.invalid || established.provider_established ||
            select_provider(established, native, 101, GetTickCount64(), {true, fg_epoch, 0}).value) return false;
        auto &fresh_ngx = make(2, provider_kind::ngx, 81, 1, 600);
        if (select_provider(established, native, 102, GetTickCount64()).value != &fresh_ngx) return false;
        establish_provider(established, fresh_ngx, 102);
        if (select_provider(established, native, 103, GetTickCount64(), {true, fg_epoch, 0}).value) return false;
        auto &renewed_fg = make(3, provider_kind::streamline, fg_epoch, 14, fg_id);
        renewed_fg.metadata.frame_generation_input = true;
        if (select_provider(established, native, 103, GetTickCount64(), {true, fg_epoch, 0}).value != &renewed_fg) return false;
      }
      for (const bool complete_before_release : {false, true}) {
        slots = {}; evaluations = {};
        auto &retired = make(0, provider_kind::ngx, 4, 1, 200);
        auto &other = make(1, provider_kind::streamline, 4, 1, 0);
        retired.command = 888; retired.retirements[0] = {native, 9};
        retired.finished = complete_before_release;
        establish_provider(established, retired, 5);
        retire_source(provider_kind::ngx, 4, 201);
        retire_source(provider_kind::ngx, 5, 200);
        retire_source(provider_kind::streamline, 4, 200);
        if (retired.invalid || !established.provider_established || established.source_id != 200 || other.invalid) return false;
        retire_source(provider_kind::ngx, 4, 200);
        if (established.provider_established || established.present_capture || established.source_id ||
            !retired.invalid || retired.failure != capture_failure::source_retired ||
            retired.command != 888 || retired.producer_fence != 1 || retired.retirements[0].value != 9 ||
            !retired.producer_submitted || other.invalid || established.admission_after != other.id) return false;
        // A result returning after explicit release cannot revive old pixels.
        finish(retired.id, true);
        if (!retired.invalid || submitted_success(retired, native) || reclaimable(retired)) return false;
        // The immediately preceding capture is exactly serial, so subtracting
        // one at release would wrongly resurrect this other provider's frame.
        if (select_provider(established, native, 6, GetTickCount64()).value) return false;
        auto &fresh = make(2, provider_kind::streamline, 4, 2, 0);
        if (fresh.id <= established.admission_after ||
            select_provider(established, native, 6, GetTickCount64()).value != &fresh) return false;
      }
      // SR and FG use distinct logical sources for the same viewport. The
      // explicit role transition removes only the superseded SR view, rather
      // than waiting for its normal age window or ignoring true multi-view.
      slots = {}; evaluations = {};
      queue_state role_owner;
      constexpr std::uint64_t fg_source = 1ull << 63;
      make(0, provider_kind::streamline, 8, 1, 0);
      auto &fg = make(1, provider_kind::streamline, 8, 2, fg_source);
      if (select_capture(role_owner, native, 7, provider_kind::streamline, GetTickCount64()).result != status::ambiguous) return false;
      fg.metadata.sequence = 3;
      begin_evaluation(8, 3, 0, provider_kind::streamline, fg_source, 0);
      if (select_capture(role_owner, native, 7, provider_kind::streamline, GetTickCount64()).value != &fg) return false;
      begin_evaluation(8, 4, 1, provider_kind::streamline, 0);
      fg.metadata.sequence = 5;
      begin_evaluation(8, 5, 0, provider_kind::streamline, fg_source, 0);
      if (select_capture(role_owner, native, 7, provider_kind::streamline, GetTickCount64()).result != status::ambiguous) return false;

      // A Present can observe an SDK nomination after its watermark advances
      // but before native validation/allocation has produced a capture slot.
      // Such an intent can retain only an established complete FG pair, never
      // select the previous depth or establish authority on a fresh queue.
      queue_state pending_owner;
      const auto prepare_pending = [&] {
        slots = {}; evaluations = {};
        pending_owner.provider_established = false;
        pending_owner.last_epoch = pending_owner.last_sequence = pending_owner.present = pending_owner.present_capture = 0;
        auto &previous = make(0, provider_kind::streamline, 20, 1, fg_source);
        previous.metadata.frame_generation_input = true;
        establish_provider(pending_owner, previous, 10);
        pending_owner.last_epoch = 20; pending_owner.last_sequence = 1;
        input candidate = previous.metadata;
        candidate.sequence = 2;
        candidate.source = std::make_shared<source_reference>();
        candidate.resource.native = 1;
        return candidate;
      };
      const auto pending_pick = [&](std::uint64_t now = GetTickCount64()) {
        return select_capture(pending_owner, native, 11, provider_kind::streamline, now);
      };
      auto candidate = prepare_pending();
      if (!begin_nomination(candidate, UINT64_MAX)) return false;
      auto pending = pending_pick();
      if (pending.value || pending.latest || !pending.pending_nomination || pending.result != status::recorded ||
          !matching_evaluation(*pending.pending_nomination, candidate)) return false;
      auto &pending_fallback = make(1, provider_kind::ngx, 60, 1, 500);
      selection_policy excluded_pending;
      excluded_pending.exclude_unconfirmed_fg = true;
      if (select_provider(pending_owner, native, 11, GetTickCount64(), excluded_pending).value != &pending_fallback) return false;
      const auto pending_decision = classify_acquisition(pending, pending_owner, {true, 20, 0});
      if (pending_decision.source_selected || !pending_decision.source_valid || !pending_decision.pending_frame ||
          pending_decision.capture_id || pending_decision.epoch != candidate.epoch ||
          pending_decision.sequence != candidate.sequence || pending_decision.source_id != candidate.source_id ||
          pending_decision.viewport != candidate.viewport) return false;
      pending_owner.provider_established = false;
      if (pending_pick().pending_nomination || pending_pick().value) return false;
      pending_owner.provider_established = true;
      if (pending_pick(pending.pending_nomination->tick + 251).pending_nomination ||
          pending_pick(pending.pending_nomination->tick + 251).result != status::stale) return false;
      end_nomination(candidate);
      if (pending_pick().pending_nomination || pending_pick().value) return false;

      // An old call returning cannot clear a later valid intent, and a missing
      // newer tag withdraws it even though no record()/finish() follows.
      if (!begin_nomination(candidate, UINT64_MAX)) return false;
      auto newer = candidate; ++newer.sequence;
      if (!begin_nomination(newer, UINT64_MAX)) return false;
      end_nomination(candidate);
      if (!pending_pick().pending_nomination || !matching_evaluation(*pending_pick().pending_nomination, newer)) return false;
      begin_evaluation(20, 4, 0, provider_kind::streamline, fg_source);
      end_nomination(newer);
      if (pending_pick().pending_nomination || pending_pick().value || pending_pick().latest) return false;

      // Exercise the real early-rejection/RAII path without fake native COM.
      candidate = prepare_pending();
      record_diagnostic rejected;
      if (nominate_evaluation(0, candidate, UINT64_MAX, &rejected) || rejected.stage != record_stage::malformed_input ||
          pending_pick().pending_nomination || pending_pick().value) return false;
      input alternatives[]{candidate, candidate, candidate};
      if (nominate_evaluation(0, alternatives, 2, UINT64_MAX, &rejected) || rejected.stage != record_stage::malformed_input ||
          pending_pick().pending_nomination || pending_pick().value) return false;
      if (!begin_nomination(candidate, UINT64_MAX)) return false;
      if (nominate_evaluation(0, alternatives, 3, UINT64_MAX, &rejected) || rejected.stage != record_stage::malformed_input ||
          pending_pick().pending_nomination || pending_pick().value) return false;
      // A batch cannot combine logical evaluations or revoke a separately
      // active nomination when its adapter contract is malformed.
      if (!begin_nomination(candidate, UINT64_MAX)) return false;
      for (unsigned mismatch = 0; mismatch != 6; ++mismatch) {
        alternatives[1] = candidate;
        if (mismatch == 0) alternatives[1].provider = provider_kind::ngx;
        if (mismatch == 1) ++alternatives[1].epoch;
        if (mismatch == 2) ++alternatives[1].sequence;
        if (mismatch == 3) ++alternatives[1].source_id;
        if (mismatch == 4) ++alternatives[1].viewport;
        if (mismatch == 5) alternatives[1].frame_generation_input = false;
        if (nominate_evaluation(0, alternatives, 2, UINT64_MAX, &rejected) || rejected.stage != record_stage::malformed_input ||
            !pending_pick().pending_nomination) return false;
      }
      if (nominate_evaluation(0, nullptr, 1, UINT64_MAX, &rejected) ||
          nominate_evaluation(0, alternatives, 0, UINT64_MAX, &rejected) ||
          nominate_evaluation(0, alternatives, 4, UINT64_MAX, &rejected) || !pending_pick().pending_nomination) return false;
      end_nomination(candidate);
      for (unsigned mismatch = 0; mismatch != 4; ++mismatch) {
        candidate = prepare_pending();
        if (mismatch == 0) ++candidate.epoch;
        if (mismatch == 1) ++candidate.source_id;
        if (mismatch == 2) ++candidate.viewport;
        if (!begin_nomination(candidate, UINT64_MAX)) return false;
        auto &current = evaluations[provider_index(candidate.provider)];
        if (mismatch == 2) current.views[0] = {}; // Isolate exact viewport identity from ambiguity.
        if (mismatch == 3) current.views[1] = {20, 2, GetTickCount64(), fg_source + 1, 1};
        if (pending_pick().pending_nomination || pending_pick().value) return false;
        end_nomination(candidate);
      }
      candidate = prepare_pending();
      if (!begin_nomination(candidate, UINT64_MAX)) return false;
      auto &published = slots[1];
      published.id = ++serial; published.metadata = candidate;
      published.source_nominated = published.nomination_only = true;
      published.pixel_failure = status::unsupported_resource;
      if (pending_pick().latest || !pending_pick().pending_nomination ||
          !pending_frame(pending_pick(), pending_owner) || pending_pick().value) return false;
      // A rejected intermediate tag does not end the enclosing batch: the next
      // API-supplied tag can still capture. Ending that scope removes the hold.
      finish(published.id, false);
      if (pending_pick().latest || pending_pick().result != status::recorded ||
          !pending_frame(pending_pick(), pending_owner)) return false;
      end_nomination(candidate);
      if (pending_pick().result != status::failed || pending_frame(pending_pick(), pending_owner)) return false;
      // A successfully tagged but unsupported texture has no pending pixels,
      // even before producer submission. It must use current-color mono.
      published = {}; published.id = ++serial; published.metadata = candidate;
      published.source_nominated = published.nomination_only = published.finished = published.success = true;
      published.pixel_failure = status::unsupported_resource;
      if (pending_pick().result != status::recorded || pending_frame(pending_pick(), pending_owner)) return false;
      published.nomination_only = false;
      if (!pending_frame(pending_pick(), pending_owner)) return false; // A real unfinished snapshot may hold.
      published.success = false;
      if (pending_frame(pending_pick(), pending_owner)) return false;

      // Preferred metadata must use its reserved capacity before taking a
      // pixel-capable slot needed by a lower-priority tag's actual capture.
      slots = {}; evaluations = {};
      begin_evaluation(candidate.epoch, candidate.sequence, candidate.viewport, candidate.provider, candidate.source_id);
      for (unsigned i = 0; i + 1 < pixel_slot_limit; ++i) {
        slots[i].id = ++serial;
        slots[i].retirement_unknown = true; // Live GPU obligations cannot be reclaimed.
      }
      const auto pixel_end = slots.begin() + pixel_slot_limit;
      for (unsigned i = pixel_slot_limit; i < slot_limit; ++i) {
        const auto ticket = record_nomination(candidate, candidate.resource.native, 0, {});
        if (!ticket || slots[i].id != ticket ||
            std::find_if(slots.begin(), pixel_end, reclaimable) != pixel_end - 1) return false;
      }
      // The reservation is a preference, not an artificial nomination limit:
      // when it is full, unused pixel capacity can still retain source authority.
      const auto overflow = record_nomination(candidate, candidate.resource.native, 0, {});
      if (!overflow || slots[pixel_slot_limit - 1].id != overflow ||
          record_nomination(candidate, candidate.resource.native, 0, {})) return false;
      finish(slots[pixel_slot_limit].id, false);
      const auto recycled = record_nomination(candidate, candidate.resource.native, 0, {});
      if (!recycled || slots[pixel_slot_limit].id != recycled ||
          slots[pixel_slot_limit - 1].id != overflow) return false;
      return true;
    }
    bool live_source_admission_regression() {
      // A pipelined game's newest nomination is still in flight at every
      // Present, so its pixels come only from the source's own newest
      // completed snapshot. Drive the production selector together with the
      // watermark, establishment and consumption bookkeeping of acquire() and
      // complete_frame(); only the producer fences are supplied by the CPU.
      struct cleanup {
        std::array<slot, slot_limit> saved_slots{slots};
        std::array<evaluation_namespace, provider_count> saved_evaluations{evaluations};
        std::array<status, provider_count> saved_attempts{attempt_status[0].load(), attempt_status[1].load()};
        bool saved_requested{requested.load()};
        status saved_reason{reason.load()};
        std::uint64_t saved_serial{serial.load()};
        ~cleanup() {
          slots = saved_slots; evaluations = saved_evaluations;
          for (unsigned i = 0; i != provider_count; ++i) attempt_status[i] = saved_attempts[i];
          requested = saved_requested; reason = saved_reason; serial = saved_serial;
        }
      } restore;
      requested = true;
      struct completion_fence final : ID3D12Fence {
        UINT64 completed{};
        HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void **out) override { if (out) *out = nullptr; return E_NOINTERFACE; }
        ULONG STDMETHODCALLTYPE AddRef() override { return 1; }
        ULONG STDMETHODCALLTYPE Release() override { return 1; }
        HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID, UINT *, void *) override { return E_NOTIMPL; }
        HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID, UINT, const void *) override { return E_NOTIMPL; }
        HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID, const IUnknown *) override { return E_NOTIMPL; }
        HRESULT STDMETHODCALLTYPE SetName(LPCWSTR) override { return E_NOTIMPL; }
        HRESULT STDMETHODCALLTYPE GetDevice(REFIID, void **out) override { if (out) *out = nullptr; return E_NOINTERFACE; }
        UINT64 STDMETHODCALLTYPE GetCompletedValue() override { return completed; }
        HRESULT STDMETHODCALLTYPE SetEventOnCompletion(UINT64, HANDLE) override { return E_NOTIMPL; }
        HRESULT STDMETHODCALLTYPE Signal(UINT64) override { return E_NOTIMPL; }
      };
      // Registered queue timelines. Selection only compares a queue's identity
      // and reads its private fence; it never calls the queue.
      struct producer_queue {
        completion_fence fence;
        std::uint64_t signaled{};
        std::unique_ptr<queue_state> *entry{};
        std::uint64_t native() const { return reinterpret_cast<std::uint64_t>(this); }
        ~producer_queue() {
          if (!entry || !*entry) return;
          (*entry)->queue.p = nullptr; (*entry)->fence.p = nullptr; entry->reset();
        }
      };
      constexpr std::uint64_t device = 41;
      // Two producer timelines (SL, NGX) and the effects queue, which owns the
      // provider selection exactly as in acquire(), so retire_source sees it.
      std::array<producer_queue, 3> producers;
      for (auto &value : producers) {
        const auto entry = std::find_if(queues.begin(), queues.end(), [](const auto &item) { return !item; });
        if (entry == queues.end()) return false;
        *entry = std::make_unique<queue_state>();
        value.entry = &*entry;
        (*entry)->queue.p = reinterpret_cast<ID3D12CommandQueue *>(&value);
        (*entry)->fence.p = &value.fence;
        (*entry)->device_identity = device;
      }
      const auto consumer = producers[2].native();
      const auto fresh_owner = [&]() -> queue_state & {
        auto &value = **producers[2].entry;
        value.provider_established = false; value.provider = provider_kind::streamline;
        value.source_id = value.last_epoch = value.last_sequence = value.last_source_tick = value.nominated_epoch = 0;
        value.present = value.present_capture = value.admission_after = 0; value.viewport = 0;
        return value;
      };
      const auto layout = [&](unsigned width, unsigned height) {
        auto value = std::make_shared<source_reference>();
        value->device_identity = device;
        value->desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        value->desc.Width = width; value->desc.Height = height; value->desc.Format = DXGI_FORMAT_R32_TYPELESS;
        return value;
      };
      struct pipeline {
        provider_kind provider{};
        std::uint64_t epoch{}, source_id{};
        source_ref source;
        producer_queue *queue{};
        unsigned first{};
        std::uint64_t sequence{};
        bool frame_generation{};
        unsigned written{};
        slot *pending{};
      };
      // The game records its next evaluation, still in flight at the next
      // Present, while its previous one completes on the GPU.
      const auto advance = [&](pipeline &p) -> slot & {
        if (p.pending) p.queue->fence.completed = std::max<UINT64>(p.queue->fence.completed, p.pending->producer_fence);
        auto &value = slots[p.first + p.written++ % 4];
        value = {};
        value.id = ++serial; value.queue = p.queue->native(); value.producer_fence = ++p.queue->signaled;
        value.texture = std::make_shared<texture_reference>();
        value.finished = value.success = value.producer_submitted = value.source_nominated = true;
        value.metadata.provider = p.provider; value.metadata.epoch = p.epoch; value.metadata.sequence = ++p.sequence;
        value.metadata.source_id = p.source_id; value.metadata.source = p.source;
        value.metadata.frame_generation_input = p.frame_generation;
        value.metadata.tick = GetTickCount64();
        begin_evaluation(p.epoch, p.sequence, 0, p.provider, p.source_id);
        p.pending = &value;
        return value;
      };
      const auto finish_gpu = [](pipeline &p) {
        p.queue->fence.completed = std::max<UINT64>(p.queue->fence.completed, p.pending->producer_fence);
      };
      // An attempt rejected before its capture slot exists (observer coverage,
      // a closed recording or render pass, proof, lifetime or identity checks):
      // the provider's sequence and view advance, its previous capture
      // completes, and only the attempt status remains.
      const auto reject_without_slot = [&](pipeline &p) {
        if (p.pending) finish_gpu(p);
        begin_evaluation(p.epoch, ++p.sequence, 0, p.provider, p.source_id);
        attempt_status[provider_index(p.provider)] = status::unsupported_state;
      };
      // Time passes for a source that stopped evaluating, including the last
      // frame the queue copied from it.
      const auto expire = [&](const pipeline &p, queue_state &queue) {
        constexpr auto age = sunshine_scene_depth::maximum_source_age_ms + 1;
        for (auto &view : evaluations[provider_index(p.provider)].views)
          if (view.source_id == p.source_id && view.tick > age) view.tick -= age;
        for (auto &value : slots)
          if (value.id && value.metadata.provider == p.provider && value.metadata.source_id == p.source_id &&
              value.metadata.tick > age) value.metadata.tick -= age;
        if (queue.provider == p.provider && queue.source_id == p.source_id && queue.last_source_tick > age)
          queue.last_source_tick -= age;
      };
      // The provider's view of each Present: acquire() classifies the pick
      // before establishment, and only a successful copy replaces the
      // retained display frame. Test snapshots own no resource, so packets
      // are described through their source allocation, as nominations are.
      acquisition_decision decided;
      retained_depth retained;
      const auto describe = [&](const slot &value, status result) {
        auto view = value; view.nomination_only = true;
        packet out; describe_packet(view, consumer, out);
        out.pixel_ready = !value.nomination_only && result == status::ready;
        return out;
      };
      // acquire() followed, for ready pixels, by a successful copy and
      // complete_frame(): the same watermark, establishment and consumption.
      // copied=false models a failed display preparation or copy: no complete_frame().
      const auto present_frame = [&](queue_state &queue, std::uint64_t present, selection_policy policy = {},
          bool copied = true) {
        const auto chosen = select_provider(queue, consumer, present, GetTickCount64(), policy);
        decided = classify_acquisition(chosen, queue, policy);
        if (queue.provider_established && chosen.result != status::ready && chosen.result != status::recorded &&
            chosen.result != status::submitted) queue.admission_after = serial.load();
        if (chosen.value) {
          establish_provider(queue, *chosen.value, present);
          if (chosen.result == status::ready && copied) {
            queue.last_epoch = chosen.value->metadata.epoch;
            queue.last_sequence = chosen.value->metadata.sequence;
            queue.last_source_tick = chosen.value->metadata.tick;
            const auto copy = describe(*chosen.value, chosen.result);
            retained = {copy.metadata, copy.capture_id, present, copy.width, copy.height, copy.format, copy.area};
          }
        }
        return chosen;
      };
      // decide_display() for the last Present against the retained frame.
      const auto display = [&](const capture_pick &chosen) {
        const auto candidate = chosen.value ? describe(*chosen.value, chosen.result) : packet{};
        return decide_display(decided, candidate, retained, {}, GetTickCount64());
      };
      const auto pending_result = [](const capture_pick &value) {
        return value.result == status::recorded || value.result == status::submitted;
      };
      const auto completed_pick = [](const capture_pick &value, const slot *expected) {
        return value.value == expected && value.result == status::ready && value.selection == selection_reason::completed_snapshot;
      };
      // One add-on epoch can number both providers; their sequences are
      // separate namespaces and must never be compared with each other.
      constexpr std::uint64_t epoch = 4;
      const auto render = layout(1280, 720);
      const auto streamline = [&](std::uint64_t source_id = 0) {
        return pipeline{provider_kind::streamline, epoch, source_id, render, &producers[0], 0, 31635};
      };
      const auto ngx = [&](std::uint64_t source_id = 9, source_ref source = {}) {
        return pipeline{provider_kind::ngx, epoch, source_id, source ? source : render, &producers[1], 8, 4686};
      };
      const auto restart = [&] {
        slots = {}; evaluations = {}; decided = {}; retained = {};
        for (auto &value : producers) { value.fence.completed = value.signaled = 0; }
      };
      // Establish a pipelined source as the delivering owner.
      const auto deliver = [&](queue_state &queue, pipeline &p, std::uint64_t &present) {
        advance(p);
        for (unsigned frame = 0; frame != 3; ++frame) {
          auto &completed = *p.pending;
          advance(p);
          if (!completed_pick(present_frame(queue, ++present), &completed)) return false;
        }
        return queue.provider_established && queue.provider == p.provider && queue.source_id == p.source_id;
      };
      std::uint64_t present = 0;

      // (a)+(f) A late SL capture took ownership and SL then stopped. Its
      // expired source must neither hide pipelined NGX nor mark a gap.
      {
        restart();
        auto &owner = fresh_owner();
        auto sl = streamline();
        auto ngx_source = ngx();
        auto &late = advance(sl); finish_gpu(sl);
        if (present_frame(owner, ++present).value != &late || owner.provider != provider_kind::streamline ||
            owner.last_sequence != late.metadata.sequence) return false;
        expire(sl, owner);
        advance(ngx_source);
        const auto first = present_frame(owner, ++present);
        if (first.latest != ngx_source.pending || !pending_result(first) || owner.admission_after) return false;
        auto &completed = *ngx_source.pending;
        advance(ngx_source);
        if (!completed_pick(present_frame(owner, ++present), &completed) || owner.provider != provider_kind::ngx) return false;
        for (unsigned frame = 0; frame != 4; ++frame) {
          auto &next = *ngx_source.pending;
          advance(ngx_source);
          if (!completed_pick(present_frame(owner, ++present), &next)) return false;
        }
        // Without an owner, the expired SL capture is still not reported in
        // front of a live NGX source that has no completed snapshot yet.
        restart();
        auto &fresh = fresh_owner();
        auto stale = streamline(); auto pending = ngx();
        advance(stale); finish_gpu(stale);
        expire(stale, fresh);
        advance(pending).producer_submitted = false; // CPU-recorded only: no source value yet.
        const auto front = select_provider(fresh, consumer, ++present, GetTickCount64());
        if (front.value || front.latest != pending.pending || front.result != status::recorded) return false;
      }
      // (b) No owner (runtime reset or new generation): a source that is only
      // ever CPU-recorded at Present establishes from its completed snapshot.
      {
        restart();
        auto &owner = fresh_owner();
        auto source = ngx();
        advance(source).producer_submitted = false;
        const auto recorded = present_frame(owner, ++present);
        if (recorded.value || recorded.result != status::recorded || owner.provider_established) return false;
        auto &completed = *source.pending; completed.producer_submitted = true;
        advance(source).producer_submitted = false;
        if (!completed_pick(present_frame(owner, ++present), &completed) || !owner.provider_established ||
            owner.provider != provider_kind::ngx || owner.source_id != 9) return false;
      }
      // (c) A DLSS quality change re-creates the NGX feature at a new render
      // resolution: a new source identity whose newest capture is in flight,
      // already submitted or still only CPU-recorded at each Present.
      for (const bool observed_release : {true, false}) for (const bool submitted_newest : {true, false}) {
        restart();
        auto &owner = fresh_owner();
        auto old_feature = ngx(9);
        if (!deliver(owner, old_feature, present)) return false;
        if (observed_release) {
          retire_source(provider_kind::ngx, epoch, 9);
          if (owner.provider_established) return false;
        } else expire(old_feature, owner); // The settings change outlasts the source age.
        auto feature = ngx(10, layout(1706, 960));
        feature.sequence = old_feature.sequence; feature.first = 12;
        advance(feature).producer_submitted = submitted_newest;
        const auto changing = present_frame(owner, ++present);
        if ((changing.value && changing.value->metadata.source_id != 10) || changing.latest != feature.pending ||
            !pending_result(changing)) return false;
        auto &completed = *feature.pending; completed.producer_submitted = true;
        advance(feature).producer_submitted = submitted_newest;
        if (!completed_pick(present_frame(owner, ++present), &completed) || owner.source_id != 10 ||
            owner.provider != provider_kind::ngx) return false;
      }
      // (d) The mirror case: NGX stops and pipelined SL takes over.
      {
        restart();
        auto &owner = fresh_owner();
        auto old_source = ngx();
        if (!deliver(owner, old_source, present)) return false;
        expire(old_source, owner);
        auto sl = streamline();
        advance(sl);
        const auto first = present_frame(owner, ++present);
        if (first.latest != sl.pending || !pending_result(first) || owner.admission_after) return false;
        auto &completed = *sl.pending;
        advance(sl);
        if (!completed_pick(present_frame(owner, ++present), &completed) || owner.provider != provider_kind::streamline) return false;
      }
      // (e) A delivering owner keeps its live source while the other provider
      // is live and readable every frame, including over a pending Present.
      for (const auto owner_kind : {provider_kind::ngx, provider_kind::streamline}) {
        restart();
        auto &owner = fresh_owner();
        auto own = owner_kind == provider_kind::ngx ? ngx() : streamline();
        auto other = owner_kind == provider_kind::ngx ? streamline() : ngx();
        if (!deliver(owner, own, present)) return false;
        for (unsigned frame = 0; frame != 6; ++frame) {
          const bool owner_advances = frame % 3 != 2; // Every third Present has no new completion.
          if (owner_advances) advance(own);
          advance(other); finish_gpu(other);
          const auto chosen = present_frame(owner, ++present);
          if ((chosen.value && chosen.value->metadata.provider != owner_kind) ||
              (owner_advances ? chosen.result != status::ready : !pending_result(chosen)) ||
              owner.provider != owner_kind || owner.admission_after) return false;
        }
        // Ownership is revocable: a failed owner attempt, or no copied owner
        // frame within the source age, lets the readable live source take over.
        auto &failed = advance(own);
        finish(failed.id, false);
        advance(other); finish_gpu(other);
        const auto failover = present_frame(owner, ++present);
        if (!failover.value || failover.value->metadata.provider == owner_kind || failover.result != status::ready) return false;
      }
      {
        restart();
        auto &owner = fresh_owner();
        auto own = ngx(); auto other = streamline();
        if (!deliver(owner, own, present)) return false;
        present_frame(owner, ++present); // Consumes nothing new: the pipeline did not advance.
        owner.last_source_tick -= sunshine_scene_depth::maximum_source_age_ms + 1;
        advance(other); finish_gpu(other);
        const auto takeover = present_frame(owner, ++present);
        if (takeover.value != other.pending || takeover.result != status::ready ||
            owner.provider != provider_kind::streamline) return false;
      }
      // A readable owner whose frames are never copied (failed display
      // preparation or copy, so no complete_frame) is not delivering: once its
      // last copied frame is older than the source age, readable SL takes over.
      {
        restart();
        auto &owner = fresh_owner();
        auto own = ngx(); auto other = streamline();
        if (!deliver(owner, own, present)) return false;
        auto &uncopied = *own.pending;
        advance(own);
        if (!completed_pick(present_frame(owner, ++present, {}, false), &uncopied)) return false;
        owner.last_source_tick -= sunshine_scene_depth::maximum_source_age_ms + 1;
        advance(own); advance(other); finish_gpu(other);
        const auto revoked = present_frame(owner, ++present);
        if (revoked.value != other.pending || revoked.result != status::ready ||
            owner.provider != provider_kind::streamline) return false;
      }
      // An owner's already copied frame is not a pending successor. A one-frame
      // stray owner therefore does not hold a readable live source of the other
      // provider for the source age.
      for (const auto owner_kind : {provider_kind::ngx, provider_kind::streamline}) {
        restart();
        auto &owner = fresh_owner();
        auto stray = owner_kind == provider_kind::ngx ? ngx() : streamline();
        auto other = owner_kind == provider_kind::ngx ? streamline() : ngx();
        advance(stray); finish_gpu(stray);
        if (present_frame(owner, ++present).value != stray.pending || owner.provider != owner_kind) return false;
        for (unsigned frame = 0; frame != 2; ++frame) {
          advance(other); finish_gpu(other);
          const auto chosen = present_frame(owner, ++present);
          if (chosen.value != other.pending || chosen.result != status::ready || owner.provider == owner_kind) return false;
        }
      }
      // The nested fallback in a pipelined game: one owner copy is rejected
      // (a metadata-only nomination, or an attempt rejected before its capture
      // slot exists), so the other provider's inner evaluation is captured and
      // is still in flight at that Present. The delivering owner keeps
      // selection and the display holds its last copied frame. The pixel-less
      // inner capture takes no authority, so the held depth is not invalidated
      // as a source change and selection does not flip to the other provider
      // and back.
      for (const auto owner_kind : {provider_kind::streamline, provider_kind::ngx})
      for (const bool slotless : {false, true}) {
        restart();
        auto &owner = fresh_owner();
        auto own = owner_kind == provider_kind::ngx ? ngx() : streamline();
        auto nested = owner_kind == provider_kind::ngx ? streamline() : ngx();
        if (!deliver(owner, own, present)) return false;
        const auto copied = retained.capture_id;
        // A pick without any capture names no source (epoch zero), so it
        // cannot contradict the retained one.
        const auto owned = [&] {
          return owner.provider_established && owner.provider == owner_kind && owner.source_id == own.source_id &&
            (!decided.epoch || (decided.provider == owner_kind && decided.source_id == own.source_id));
        };
        const auto holds = [&](const capture_pick &chosen) {
          const auto update = display(chosen);
          return update.action == display_action::hold && retained.capture_id == copied;
        };
        slot *rejected{};
        if (slotless) reject_without_slot(own);
        else {
          rejected = &advance(own);
          rejected->nomination_only = true; rejected->pixel_failure = status::unsupported_resource;
        }
        advance(nested);
        const auto held = present_frame(owner, ++present);
        if (held.value != rejected || (slotless && (held.latest || held.pending_nomination)) ||
            held.result != (slotless ? status::unsupported_state : status::unsupported_resource) ||
            !owned() || !holds(held)) return false;
        // The owner resumes; the inner evaluation stops and its capture completes.
        finish_gpu(nested);
        auto &resumed = advance(own);
        const auto pending_owner = present_frame(owner, ++present);
        if (pending_owner.result == status::ready || !owned() || !holds(pending_owner)) return false;
        advance(own);
        const auto fresh = present_frame(owner, ++present);
        if (!completed_pick(fresh, &resumed) || !owned() || display(fresh).action != display_action::copy_fresh ||
            retained.capture_id != resumed.id) return false;
      }
      // A live owner whose attempts fail (rejected nomination-only copies,
      // failed evaluations, a concurrently evaluated second view, or attempts
      // rejected before a capture slot exists) keeps selection only while it
      // is delivering. Once its last copied frame is
      // older than the source age, it holds nothing in front of a pipelined
      // source of the other provider: returning the failing owner would mark a
      // gap at every Present and keep the other source's completed snapshots
      // unreadable. The pending source is reported instead (its submitted
      // nomination may carry source authority, never pixels), so its first
      // completed snapshot after the gap is selected, whether it started while
      // the owner was still delivering or only after it stopped.
      for (const auto owner_kind : {provider_kind::streamline, provider_kind::ngx})
      for (unsigned failure = 0; failure != 4; ++failure)
      for (const bool other_first : {true, false}) {
        restart();
        auto &owner = fresh_owner();
        auto own = owner_kind == provider_kind::ngx ? ngx() : streamline();
        auto other = owner_kind == provider_kind::ngx ? streamline() : ngx();
        auto second_view = owner_kind == provider_kind::ngx ? ngx(11) : streamline(2);
        second_view.first = 16;
        if (!deliver(owner, own, present)) return false;
        const auto fail = [&] {
          if (failure == 3) { reject_without_slot(own); return; }
          auto &value = advance(own);
          if (failure == 0) { value.nomination_only = true; value.pixel_failure = status::unsupported_resource; }
          else if (failure == 1) finish(value.id, false);
          else { second_view.sequence = own.sequence; advance(second_view); own.sequence = second_view.sequence; }
        };
        // While delivering, the failing owner keeps selection and marks the gap.
        // A slot-less owner pick names no capture; the other provider's pick
        // would name its pending one.
        for (unsigned frame = 0; frame != 2; ++frame) {
          fail();
          if (other_first) advance(other);
          const auto kept = present_frame(owner, ++present);
          const auto *source = kept.value ? kept.value : kept.latest;
          if ((source ? source->metadata.provider != owner_kind : failure != 3 || kept.pending_nomination) ||
              kept.result == status::ready || pending_result(kept) ||
              owner.provider != owner_kind || !owner.admission_after) return false;
        }
        // No owner frame was copied within the source age.
        owner.last_source_tick -= sunshine_scene_depth::maximum_source_age_ms + 1;
        const auto gap = owner.admission_after;
        fail(); advance(other);
        const auto handover = present_frame(owner, ++present);
        if ((handover.value && handover.value != other.pending) || handover.latest != other.pending ||
            !pending_result(handover) || owner.admission_after != gap) return false;
        for (unsigned frame = 0; frame != 2; ++frame) {
          auto &completed = *other.pending;
          fail(); advance(other);
          if (!completed_pick(present_frame(owner, ++present), &completed) ||
              owner.provider == owner_kind || owner.admission_after != gap) return false;
        }
      }
      // Required FG keeps its mandatory SL scope: a pipelined FG source uses
      // its own completed snapshot, and NGX is never substituted meanwhile.
      {
        restart();
        auto &owner = fresh_owner();
        auto fg = streamline((1ull << 63) | 0); fg.frame_generation = true;
        auto readable_ngx = ngx();
        const selection_policy policy{true, epoch, 0};
        advance(fg);
        advance(readable_ngx); finish_gpu(readable_ngx);
        const auto waiting = present_frame(owner, ++present, policy);
        if ((waiting.value && !waiting.value->metadata.frame_generation_input) || waiting.latest != fg.pending ||
            !pending_result(waiting)) return false;
        auto &completed = *fg.pending;
        advance(fg);
        if (!completed_pick(present_frame(owner, ++present, policy), &completed) || owner.provider != provider_kind::streamline) return false;
      }
      return true;
    }
    namespace {
      struct private_object final : ID3D12Object {
        ULONG references{1};
        std::uint64_t cookie{};
        unsigned writes{};
        bool refuse_write{};
        HRESULT read_failure{S_OK};
        struct bytes_entry { GUID key{}; std::uint64_t value{}; bool present{}; };
        std::array<bytes_entry,4> private_bytes{};
        GUID interface_key{};
        IUnknown *private_interface{};
        ~private_object() { if (private_interface) private_interface->Release(); }
        HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **out) override {
          if (!out) return E_POINTER;
          *out = nullptr;
          if (IsEqualIID(iid, IID_IUnknown) || IsEqualIID(iid, IID_ID3D12Object)) { *out = this; AddRef(); return S_OK; }
          return E_NOINTERFACE;
        }
        ULONG STDMETHODCALLTYPE AddRef() override { return ++references; }
        ULONG STDMETHODCALLTYPE Release() override { return --references; }
        HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID key, UINT *size, void *out) override {
          if (FAILED(read_failure)) return read_failure;
          if (!size) return E_POINTER;
          if (private_interface && IsEqualGUID(key, interface_key)) {
            if (!out || *size < sizeof(private_interface)) { *size = sizeof(private_interface); return DXGI_ERROR_MORE_DATA; }
            *size = sizeof(private_interface); private_interface->AddRef();
            *static_cast<IUnknown **>(out) = private_interface; return S_OK;
          }
          const std::uint64_t *value = IsEqualGUID(key, source_guid) && cookie ? &cookie : nullptr;
          for (const auto &entry : private_bytes) if (entry.present && IsEqualGUID(key, entry.key)) value = &entry.value;
          if (!value) return DXGI_ERROR_NOT_FOUND;
          if (!out || *size < sizeof(*value)) { *size = sizeof(*value); return DXGI_ERROR_MORE_DATA; }
          *size = sizeof(*value); *static_cast<std::uint64_t *>(out) = *value; return S_OK;
        }
        HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID key, UINT size, const void *data) override {
          if (refuse_write) return E_FAIL;
          if (size != sizeof(cookie) || !data) return E_INVALIDARG;
          if (IsEqualGUID(key, source_guid)) { cookie = *static_cast<const std::uint64_t *>(data); ++writes; return S_OK; }
          for (auto &entry : private_bytes) if (!entry.present || IsEqualGUID(key, entry.key)) {
            entry = {key, *static_cast<const std::uint64_t *>(data), true}; return S_OK;
          }
          return E_FAIL;
        }
        HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID key, const IUnknown *data) override {
          if (refuse_write || (private_interface && !IsEqualGUID(key, interface_key))) return E_FAIL;
          auto *replacement = const_cast<IUnknown *>(data);
          if (replacement) replacement->AddRef();
          if (private_interface) private_interface->Release();
          private_interface = replacement; interface_key = key;
          return S_OK;
        }
        HRESULT STDMETHODCALLTYPE SetName(LPCWSTR) override { return E_NOTIMPL; }
      };
      std::uint64_t native_object(private_object &value) { return reinterpret_cast<std::uint64_t>(&value); }
      bool retired_recording_regression() {
        auto producer = std::make_shared<sunshine_native_command::recording_lifetime>();
        auto consumer = std::make_shared<sunshine_native_command::recording_lifetime>();
        producer->cookie = 11; consumer->cookie = 22;
        slot submitted;
        submitted.id = 1; submitted.command = 11; submitted.producer_recording = producer;
        submitted.producer_submitted = submitted.acquired = submitted.retirement_unknown = true;
        submitted.producer_fence = 7; submitted.retirements[0] = {77, 13};
        submitted.consumers[0] = 22; submitted.consumer_recordings[0] = consumer;
        retire_dead_recordings(submitted);
        if (submitted.command != 11 || submitted.consumers[0] != 22) return false;
        producer->cookie = consumer->cookie = 0;
        retire_dead_recordings(submitted);
        if (submitted.command || submitted.producer_recording || submitted.consumers[0] || submitted.consumer_recordings[0] ||
            submitted.invalid || !submitted.producer_submitted || !submitted.retirement_unknown ||
            submitted.producer_fence != 7 || submitted.retirements[0].value != 13 || reclaimable(submitted)) return false;
        slot abandoned;
        abandoned.id = 2; abandoned.command = 11; abandoned.producer_recording = producer;
        retire_dead_recordings(abandoned);
        if (abandoned.command || !abandoned.invalid || !abandoned.finished || abandoned.producer_submitted) return false;
        // A discarded CPU consumer can retire its recording reference, but the
        // producer fence is still required and is never advanced by destruction.
        slot consumer_only;
        consumer_only.id = 3; consumer_only.producer_submitted = consumer_only.acquired = true;
        consumer_only.producer_fence = 17;
        consumer_only.consumers[0] = 22; consumer_only.consumer_recordings[0] = consumer;
        retire_dead_recordings(consumer_only);
        return !consumer_only.consumers[0] && !consumer_only.consumer_recordings[0] && !consumer_only.invalid &&
          consumer_only.producer_submitted && consumer_only.producer_fence == 17 &&
          std::none_of(consumer_only.retirements.begin(), consumer_only.retirements.end(), [](const auto &point) { return point.queue != 0; });
      }
      bool command_population_regression() {
        const auto baseline = command_storage::live_count();
        std::array<std::unique_ptr<private_object>,160> objects;
        std::array<recording_ref,160> lives;
        for (unsigned i=0; i!=objects.size(); ++i) {
          objects[i] = std::make_unique<private_object>();
          const auto cookie = ++serial;
          if (!native_observer::set_recording_cookie(native_object(*objects[i]), cookie)) return false;
          auto owner = command(native_object(*objects[i]), cookie, true);
          if (!owner || owner->invalid || owner->observation_generation != command_generation) return false;
          lives[i] = owner.life();
          close(native_object(*objects[i]), cookie, S_OK);
          if (!owner->closed || lives[i]->cookie.load() != cookie) return false;
        }
        if (command_storage::live_count() != baseline + objects.size()) return false;
        for (unsigned i=0; i!=objects.size(); ++i) {
          const auto retired = lives[i];
          const auto old_cookie = retired->cookie.load();
          slot submitted;
          submitted.id = 1; submitted.command = old_cookie; submitted.producer_recording = retired;
          submitted.producer_submitted = true; submitted.producer_fence = 7;
          objects[i].reset(); // Native private-data teardown; no explicit destruction callback.
          if (retired->cookie.load() != 0 || command_storage::live_count() != baseline + objects.size() - 1) return false;
          retire_dead_recordings(submitted);
          if (submitted.command || submitted.invalid || !submitted.producer_submitted || submitted.producer_fence != 7) return false;
          objects[i] = std::make_unique<private_object>();
          const auto cookie = ++serial;
          auto owner = command(native_object(*objects[i]), cookie, true);
          if (!owner || owner->invalid || cookie == old_cookie || retired->cookie.load() != 0) return false;
          lives[i] = owner.life();
        }
        for (auto &object : objects) object.reset();
        return command_storage::live_count() == baseline &&
          std::all_of(lives.begin(), lives.end(), [](const auto &life) { return life->cookie.load() == 0; });
      }
    }
    bool initial_recording_regression() {
      private_object native, resource;
      const auto address = native_object(native);
      if (source_cookie(&resource) || native_observer::get_recording_cookie(address) ||
          !native_observer::recording_cookie_absent(address)) return false;
      // These are the production entry and post-call callbacks for a newly
      // created, already-open list. There has been no Reset or SDK nomination.
      std::uint64_t cookie{};
      associate_recording(address, &cookie);
      auto owner = command(address, cookie, false);
      if (!cookie || !owner || owner->closed || owner->invalid || !owner->states.empty() ||
          native_observer::get_recording_cookie(address) != cookie || owner.life()->cookie.load() != cookie) return false;
      D3D12_RESOURCE_BARRIER transition{};
      transition.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      transition.Transition.pResource = reinterpret_cast<ID3D12Resource *>(&resource);
      transition.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
      transition.Transition.StateAfter = sampled_state;
      // A list ReShade never reported cannot admit a capture: nothing is tracked.
      barriers(address, cookie, 1, &transition);
      if (source_cookie(&resource) || !owner->states.empty() || owner->native_barrier_calls) return false;
      owner->reported = true; // ReShade's create event.
      barriers(address, cookie, 1, &transition);
      const auto source = source_cookie(&resource);
      const auto *observed = state(*owner, source, false);
      if (!source || resource.writes != 1 || !observed || !observed->known || observed->blocked ||
          observed->value != sampled_state || retain_source_cookie(&resource) != source || resource.writes != 1) return false;
      if (owner->native_barrier_calls != 1 || owner->native_transition_count != 1 || owner->last_barrier_command != address) return false;
      private_object other_command;
      const auto other = command(native_object(other_command), ++serial, true);
      if (!other || state(*other, source, false)) return false; // Another list's barrier is never a state proof.
      // Entry association never invents a missing resource state or overwrites
      // an already associated recording, including malformed zero data.
      std::uint64_t duplicate = 99;
      associate_recording(address, &duplicate);
      if (duplicate || owner->cookie != cookie) return false;
      close(address, cookie, S_OK);
      if (!owner->closed) return false;
      reset(address, cookie, S_OK, true);
      const auto next_cookie = native_observer::get_recording_cookie(address);
      if (!next_cookie || next_cookie == cookie || owner->closed || !owner->states.empty() ||
          owner->native_barrier_calls || owner->native_transition_count || owner->last_barrier_command) return false;
      // A post-call callback carrying the entry's old or zero identity cannot
      // attach to the new recording after Reset. This applies to state, close,
      // render-pass and enhanced-barrier evidence alike.
      const native_observer::enhanced_texture enhanced_texture{reinterpret_cast<ID3D12Resource *>(&resource),
        D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE, true};
      for (const auto stale : {std::uint64_t{}, cookie}) {
        barriers(address, stale, 1, &transition);
        close(address, stale, S_OK);
        render_pass(address, stale, true);
        enhanced_textures(address, stale, 1, &enhanced_texture);
      }
      if (owner->closed || owner->render_pass || owner->invalid || !owner->states.empty()) return false;
      render_pass(address, next_cookie, true);
      if (!owner->render_pass) return false;
      render_pass(address, next_cookie, false);
      // An enhanced texture barrier into a layout with a legacy equivalent sets
      // the texture's state; one without (queue-specific) blocks only it.
      const native_observer::enhanced_texture shader_resource{reinterpret_cast<ID3D12Resource *>(&resource),
        D3D12_BARRIER_LAYOUT_SHADER_RESOURCE, true};
      enhanced_textures(address, next_cookie, 1, &shader_resource);
      const auto *mapped = state(*owner, source_cookie(&resource), false);
      if (owner->render_pass || owner->invalid || !mapped || mapped->blocked || !mapped->known ||
          mapped->value != (D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE))
        return false;
      const native_observer::enhanced_texture stencil_only{reinterpret_cast<ID3D12Resource *>(&resource),
        D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE, false};
      enhanced_textures(address, next_cookie, 1, &stencil_only);
      if (mapped->blocked) return false;
      enhanced_textures(address, next_cookie, 1, &enhanced_texture);
      const auto *enhanced = state(*owner, source_cookie(&resource), false);
      if (owner->render_pass || owner->invalid || !enhanced || !enhanced->blocked) return false;

      private_object closed, pass, malformed, refused, orphan;
      std::uint64_t first{};
      associate_recording(native_object(closed), &first); close(native_object(closed), first, S_OK);
      auto closed_owner = command(native_object(closed), first, false);
      if (!first || !closed_owner || !closed_owner->closed) return false;
      associate_recording(native_object(pass), &first); render_pass(native_object(pass), first, true);
      auto pass_owner = command(native_object(pass), first, false);
      if (!first || !pass_owner || !pass_owner->render_pass) return false;
      if (!native_observer::set_recording_cookie(native_object(malformed), 0) ||
          native_observer::recording_cookie_absent(native_object(malformed))) return false;
      associate_recording(native_object(malformed), &first);
      if (first || command_storage::acquire(&malformed, false)) return false;
      refused.refuse_write = true;
      associate_recording(native_object(refused), &first);
      if (first || native_observer::get_recording_cookie(native_object(refused))) return false;
      auto orphan_owner = command(native_object(orphan), ++serial, true);
      associate_recording(native_object(orphan), &first);
      return orphan_owner && !first && !native_observer::get_recording_cookie(native_object(orphan));
    }
    bool recording_recovery_regression() {
      // Exercise actual global loss and allocation together. Existing named
      // recordings remain rejected; a distinct new recording starts clean.
      struct cleanup {
        std::array<slot, slot_limit> saved_slots{slots};
        status saved_reason{reason.load()};
        std::uint64_t saved_generation{command_generation};
        ~cleanup() { slots = saved_slots; reason = saved_reason; command_generation = saved_generation; }
      } restore;
      private_object old_object, fresh_object;
      const auto old_cookie = ++serial;
      auto old = command(native_object(old_object), old_cookie, true);
      if (!old) return false;
      old->closed = old->render_pass = true;
      old->states.push_back({++serial, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true});
      invalidate_all();
      auto fresh = command(native_object(fresh_object), ++serial, true);
      if (!fresh || &*fresh == &*old || fresh->invalid || fresh->closed || fresh->render_pass ||
          std::any_of(fresh->states.begin(), fresh->states.end(), [](const auto &v) { return v.source || v.value || v.known || v.blocked; })) return false;
      auto lost = command(native_object(old_object), old_cookie, true);
      if (!lost || &*lost != &*old || !old->invalid || !old->closed || !old->render_pass || !old->states[0].known) return false;
      reset(native_object(old_object), old_cookie, S_OK);
      const auto new_cookie = native_observer::get_recording_cookie(native_object(old_object));
      auto reused = command(native_object(old_object), new_cookie, true);
      return new_cookie && new_cookie != old_cookie && reused && &*reused == &*old &&
        !command(native_object(old_object), old_cookie, false) && !reused->invalid && !reused->closed && !reused->render_pass &&
        reused.life()->cookie.load() == new_cookie &&
        std::none_of(reused->states.begin(), reused->states.end(), [](const auto &v) { return v.source || v.value || v.known || v.blocked; }) &&
        command_population_regression() && retired_recording_regression();
    }
    bool recording_state_growth_regression() {
      private_object command_object, target;
      std::array<private_object, 160> unrelated;
      const auto native_command = native_object(command_object);
      const auto cookie = ++serial;
      auto recording = command(native_command, cookie, true);
      if (!recording) return false;
      recording->reported = true; // ReShade's create event.
      const auto target_id = retain_source_cookie(&target);
      D3D12_RESOURCE_BARRIER transition{};
      transition.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      transition.Transition.pResource = reinterpret_cast<ID3D12Resource *>(&target);
      transition.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
      barriers(native_command, cookie, 1, &transition);
      const auto append_unrelated = [&](unsigned first, unsigned end) {
        for (unsigned i = first; i != end; ++i) {
          D3D12_RESOURCE_BARRIER split = transition;
          split.Flags = D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY;
          split.Transition.pResource = reinterpret_cast<ID3D12Resource *>(&unrelated[i]);
          barriers(native_command, cookie, 1, &split);
          const auto *stored = state(*recording, unrelated[i].cookie, false);
          if (recording->invalid || !stored || !stored->blocked) return false;
        }
        return true;
      };
      if (!append_unrelated(0, 80) || recording->states.size() != 81) return false;
      // Unrelated barriers beyond the old cap must leave the target's capture
      // admission facts intact: open, valid recording and known nonzero state.
      const auto *selected = state(*recording, target_id, false);
      if (recording->invalid || recording->closed || recording->render_pass || !selected ||
          !selected->known || selected->blocked || selected->value != transition.Transition.StateAfter) return false;
      // A split barrier blocks the target until Reset, across a later ordinary
      // transition and further storage growth.
      auto split = transition;
      split.Flags = D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY;
      barriers(native_command, cookie, 1, &split);
      if (!append_unrelated(80, unsigned(unrelated.size()))) return false;
      barriers(native_command, cookie, 1, &transition);
      selected = state(*recording, target_id, false);
      if (recording->invalid || !selected || !selected->blocked || recording->states.size() != 161) return false;
      const auto capacity = recording->states.capacity();
      const auto *storage = recording->states.data();
      reset(native_command, cookie, S_OK, true);
      const auto next_cookie = native_observer::get_recording_cookie(native_command);
      if (!next_cookie || next_cookie == cookie || recording->cookie != next_cookie || recording->invalid ||
          !recording->states.empty() || recording->states.capacity() != capacity || recording->states.data() != storage) return false;
      barriers(native_command, next_cookie, 1, &transition);
      selected = state(*recording, target_id, false);
      return selected && selected->known && !selected->blocked && !recording->invalid &&
        recording->states.capacity() == capacity && recording->states.data() == storage;
    }
    bool recording_state_loss_regression() {
      // Call the production observer callback before any source cookie exists.
      // Private-data storage is a COM stub; no native API/GPU work is issued.
      private_object native, unrelated, alias_peer, refused, command_object;
      const auto native_command = native_object(command_object);
      const auto cookie = ++serial;
      auto recording = command(native_command, cookie, true);
      auto capture = std::find_if(slots.begin(), slots.end(), [](const auto &v) { return !v.id; });
      if (!recording || capture == slots.end()) return false;
      struct cleanup {
        slot *capture;
        ~cleanup() { *capture = {}; }
      } restore{&*capture};
      const auto restart_case = [&] {
        recording->restart(cookie, command_generation);
        recording->reported = true; // ReShade's lifecycle drives this recording.
        recording.life()->cookie.store(cookie);
      };
      recording->reported = true;
      *capture = {}; // Match record() allocation after a prior global invalidation.
      capture->id = ++serial; capture->command = cookie; capture->finished = capture->success = capture->producer_submitted = true;
      D3D12_RESOURCE_BARRIER value{};
      value.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      // All calls made here use ID3D12Object's inherited private-data methods.
      value.Transition.pResource = reinterpret_cast<ID3D12Resource *>(&native);
      value.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      barriers(native_command, cookie, 1, &value);
      const auto *ordinary = state(*recording, native.cookie, false);
      if (recording->invalid || !native.cookie || !ordinary || !ordinary->known || ordinary->blocked ||
          ordinary->value != D3D12_RESOURCE_STATE_UNORDERED_ACCESS || capture->invalid) return false;
      // Pre-nomination identity is not permission to invent another plane's
      // state, or to reinterpret unsupported native states as ordinary ones.
      private_object different_plane;
      value.Transition.pResource = reinterpret_cast<ID3D12Resource *>(&different_plane);
      value.Transition.Subresource = 1;
      barriers(native_command, cookie, 1, &value);
      if (different_plane.cookie || recording->invalid) return false;
      value.Transition.Subresource = 0;
      value.Transition.StateAfter = static_cast<D3D12_RESOURCE_STATES>(0x80000000u);
      barriers(native_command, cookie, 1, &value);
      const auto *unsupported = state(*recording, different_plane.cookie, false);
      if (!different_plane.cookie || !unsupported || unsupported->known || recording->invalid) return false;
      value.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      for (const auto flags : {D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY, D3D12_RESOURCE_BARRIER_FLAG_END_ONLY}) {
        restart_case();
        private_object pretag;
        value.Transition.pResource = reinterpret_cast<ID3D12Resource *>(&pretag);
        value.Flags = flags;
        barriers(native_command, cookie, 1, &value);
        const auto *blocked = state(*recording, pretag.cookie, false);
        if (recording->invalid || !pretag.cookie || !blocked || !blocked->blocked || capture->invalid) return false;
        value.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barriers(native_command, cookie, 1, &value);
        if (!blocked->blocked || recording->invalid || capture->invalid) return false;
      }
      // Unrelated named exceptional barriers must leave selected depth usable.
      restart_case();
      const auto selected = retain_source_cookie(&native);
      value.Transition.pResource = reinterpret_cast<ID3D12Resource *>(&native);
      barriers(native_command, cookie, 1, &value);
      auto *selected_state = state(*recording, selected, false);
      if (!selected_state || !selected_state->known || selected_state->blocked) return false;
      value.Flags = D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY;
      value.Transition.pResource = reinterpret_cast<ID3D12Resource *>(&unrelated);
      barriers(native_command, cookie, 1, &value);
      // An alias changes no resource state: it neither registers nor blocks
      // its resources, and the unrelated split keeps its own block.
      value = {}; value.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
      value.Aliasing.pResourceBefore = reinterpret_cast<ID3D12Resource *>(&unrelated);
      value.Aliasing.pResourceAfter = reinterpret_cast<ID3D12Resource *>(&alias_peer);
      barriers(native_command, cookie, 1, &value);
      const auto *unrelated_state = state(*recording, unrelated.cookie, false);
      selected_state = state(*recording, selected, false);
      if (recording->invalid || selected_state->blocked || !selected_state->known || !unrelated_state ||
          !unrelated_state->blocked || alias_peer.cookie || capture->invalid) return false;
      // Even a named alias of selected depth keeps its observed state: the SDK
      // consumes the source where it is copied, so the game keeps it active there.
      value.Aliasing.pResourceBefore = reinterpret_cast<ID3D12Resource *>(&native);
      barriers(native_command, cookie, 1, &value);
      value = {}; value.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      value.Transition.pResource = reinterpret_cast<ID3D12Resource *>(&native);
      value.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      barriers(native_command, cookie, 1, &value);
      selected_state = state(*recording, selected, false);
      if (recording->invalid || !selected_state || selected_state->blocked || !selected_state->known ||
          selected_state->value != D3D12_RESOURCE_STATE_UNORDERED_ACCESS || capture->invalid) return false;
      // Wildcard aliases (NULL before, after or both) likewise leave the recording intact.
      for (unsigned wildcard = 0; wildcard != 3; ++wildcard) {
        restart_case();
        value = {}; value.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
        if (wildcard == 1) value.Aliasing.pResourceBefore = reinterpret_cast<ID3D12Resource *>(&native);
        if (wildcard == 2) value.Aliasing.pResourceAfter = reinterpret_cast<ID3D12Resource *>(&native);
        barriers(native_command, cookie, 1, &value);
        if (recording->invalid || capture->invalid) return false;
      }
      restart_case();
      refused.refuse_write = true;
      value = {}; value.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      value.Transition.pResource = reinterpret_cast<ID3D12Resource *>(&refused);
      barriers(native_command, cookie, 1, &value);
      if (!recording->invalid || recording->invalidation != recording_loss::source_identity_unavailable || capture->invalid) return false;
      // Only exact private-data absence permits first-use association.
      private_object malformed, failed_read;
      malformed.private_bytes[0] = {source_guid, 0, true};
      failed_read.read_failure = E_FAIL;
      for (auto *unavailable : {&malformed, &failed_read}) {
        restart_case();
        value.Transition.pResource = reinterpret_cast<ID3D12Resource *>(unavailable);
        barriers(native_command, cookie, 1, &value);
        if (!recording->invalid || recording->invalidation != recording_loss::source_identity_unavailable ||
            unavailable->writes || capture->invalid) return false;
      }
      restart_case();
      value.Flags = D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY;
      value.Transition.pResource = reinterpret_cast<ID3D12Resource *>(&refused);
      barriers(native_command, cookie, 1, &value);
      if (!recording->invalid || recording->invalidation != recording_loss::source_identity_unavailable || capture->invalid) return false;
      restart_case();
      // Exercise the real allocation-failure boundary without exhausting the
      // test process. Growth success is covered below; failure still closes.
      while (recording->states.size() != recording->states.capacity())
        recording->states.push_back({++serial, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true});
      struct restore_growth { ~restore_growth() { fail_state_growth = false; } } restore_failure;
      fail_state_growth = true;
      if (!recording->states.empty() && !state(*recording, recording->states.front().source, true)) return false;
      value.Transition.pResource = reinterpret_cast<ID3D12Resource *>(&native);
      barriers(native_command, cookie, 1, &value);
      if (!recording->invalid || recording->invalidation != recording_loss::source_state_capacity || capture->invalid) return false;
      fail_state_growth = false;
      reset(native_command, cookie, S_OK, true);
      const auto new_cookie = native_observer::get_recording_cookie(native_command);
      return new_cookie && new_cookie != cookie && recording->cookie == new_cookie &&
        recording.life()->cookie.load() == new_cookie && !recording->invalid && !capture->invalid && capture->command == 0 &&
        std::none_of(recording->states.begin(), recording->states.end(), [](const auto &v) { return v.blocked || v.source; }) &&
        recording_state_growth_regression();
    }
    bool source_cookie_reentry_regression() {
      // Real production cookie allocation and command-state lookup, with only
      // ID3D12Object private-data storage mocked. No GPU or source state is faked
      // through the public capture path.
      private_object native, other, refused;
      auto first = std::make_shared<source_reference>();
      first->cookie = retain_source_cookie(&native);
      const auto identity = first->cookie;
      if (!identity || native.writes != 1 ||
          sunshine_native_identity::resource_cookie(&native, true) != identity || native.writes != 1) return false;
      command_state recording;
      auto *before = state(recording, identity, true);
      if (!before) return false;
      before->known = true; before->value = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      std::weak_ptr<source_reference> weak = first;
      first.reset(); // Metadata expires; the native resource remains alive.
      if (!weak.expired()) return false;
      auto returned = std::make_shared<source_reference>();
      returned->cookie = retain_source_cookie(&native);
      const auto *after = state(recording, returned->cookie, false);
      if (returned->cookie != identity || native.writes != 1 || after != before || !after->known ||
          after->value != D3D12_RESOURCE_STATE_UNORDERED_ACCESS) return false;
      const auto fallback = sunshine_native_identity::allocate_identity();
      const auto replacement = retain_source_cookie(&other);
      const auto backup = sunshine_native_identity::allocate_identity();
      refused.refuse_write = true;
      return fallback > identity && replacement > fallback && backup > replacement &&
        sunshine_native_identity::resource_cookie(&native, true) == identity && native.writes == 1 &&
        !state(recording, replacement, false) &&
        retain_source_cookie(&refused) == 0 && refused.cookie == 0;
    }
    bool unsupported_com_boundary_regression() {
      // Only the three IUnknown slots exist. Reaching GetPrivateData/GetDevice
      // or a graphics-list slot without successful QI is a real invalid access.
      struct unknown_only final : IUnknown {
        ULONG references = 1; unsigned queries = 0;
        HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **out) override {
          ++queries;
          if (!out) return E_POINTER;
          *out = nullptr;
          if (IsEqualIID(iid, IID_IUnknown)) { *out = this; AddRef(); return S_OK; }
          return E_NOINTERFACE;
        }
        ULONG STDMETHODCALLTYPE AddRef() override { return ++references; }
        ULONG STDMETHODCALLTYPE Release() override { return --references; }
      } object;
      const auto native = reinterpret_cast<std::uint64_t>(&object);
      initialize(true);
      observe_command(native); observe_queue(native); command_destroyed(native);
      input value;
      value.epoch = value.sequence = 1;
      value.source = std::make_shared<source_reference>();
      value.projection = {0.0, .0625, true, true};
      value.resource.native = native;
      value.proof = sunshine_scene_depth::state_proof::declared;
      value.native_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      value.valid_until = sunshine_scene_depth::lifetime::until_present;
      record_diagnostic diagnostic;
      const bool rejected_record = record(native, value, &diagnostic) == 0 && last_status() == status::unsupported_queue &&
        diagnostic.result == status::unsupported_queue && diagnostic.stage == record_stage::command_interface && diagnostic.command == native;
      packet consumer; consumer.ownership = std::make_shared<texture_reference>();
      consumer_diagnostic consumer_diagnostics;
      const auto prior_queries = object.queries;
      const bool rejected_missing_pixels = !mark_consumer(native, consumer, &consumer_diagnostics) &&
        consumer_diagnostics.result == consumer_status::capture_not_ready && object.queries == prior_queries;
      consumer.pixel_ready = true; // Exercise exact QI rejection, not the earlier readiness gate.
      const bool rejected_consumer = !mark_consumer(native, consumer);
      const bool rejected_owner = !provider_active(native);
      SetLastError(731);
      const bool rejected_identity = !sunshine_native_identity::identify_resource(native) && GetLastError() == 731;
      shutdown();
      return rejected_record && rejected_missing_pixels && rejected_consumer && rejected_owner && rejected_identity &&
        object.queries == 7 && object.references == 1;
    }
    bool zero_cookie_submission_regression() {
      std::array<slot, slot_limit> captures;
      captures[0].id = 1; captures[0].producer_submitted = true; // Reset producer; still owned/in flight.
      captures[0].consumers[0] = 44;
      captures[1].id = 2; captures[1].command = 55;
      std::array<bool, slot_limit> producers{}, consumers{};
      const native_observer::command_identity unknown[]{{111, 0}, {222, 0}};
      if (identify_submissions(captures, 2, unknown, producers, consumers) || captures[0].invalid ||
          std::any_of(producers.begin(), producers.end(), [](bool v) { return v; }) ||
          std::any_of(consumers.begin(), consumers.end(), [](bool v) { return v; })) return false;
      const native_observer::command_identity mixed[]{{111, 0}, {333, 55}, {444, 44}};
      if (!identify_submissions(captures, 3, mixed, producers, consumers) || captures[0].invalid ||
          producers[0] || !producers[1] || !consumers[0] || consumers[1]) return false;
      // A real repeated recording must still be rejected as a stale replay.
      captures[1].producer_submitted = true;
      identify_submissions(captures, 3, mixed, producers, consumers);
      return captures[1].invalid && !captures[0].invalid;
    }
    bool cross_queue_completion_regression() {
      // Exercise the same admission and retirement functions used by live
      // acquire/submitted/reclaimable, supplying only native fence results.
      constexpr std::uint64_t producer = 71, consumer = 72, other = 73, device = 41;
      slot value;
      value.id = 1; value.queue = producer; value.producer_fence = 500;
      value.finished = value.success = value.producer_submitted = true;
      const auto admission = [&](std::uint64_t native, std::uint64_t identity, queue_progress progress) {
        return submission_status(value, native, identity, progress);
      };
      if (admission(producer, device, {}) != status::ready ||
          admission(producer, device, {device, UINT64_MAX, true}) != status::ready ||
          admission(consumer, device, {device, 499, true}) != status::submitted ||
          admission(consumer, device, {device, 500, true}) != status::ready ||
          admission(consumer, device, {device, 900, true}) != status::ready ||
          admission(consumer, device + 1, {device, 900, true}) != status::unsupported_queue ||
          admission(consumer, device, {device, UINT64_MAX, true}) != status::failed ||
          admission(consumer, device, {device, 900, false}) != status::failed) return false;
      value.command = 99;
      value.producer_recording = std::make_shared<sunshine_native_command::recording_lifetime>();
      value.producer_recording->cookie = 99;
      if (admission(consumer, device, {device, 900, true}) != status::submitted ||
          admission(producer, device, {}) != status::ready) return false;
      value.producer_recording->cookie = 0; // Destruction retires the exact old recording.
      if (admission(consumer, device, {device, 900, true}) != status::ready) return false;
      value.producer_recording->cookie = 100; // Reset starts a different recording.
      if (admission(consumer, device, {device, 900, true}) != status::ready) return false;
      value.producer_recording.reset();
      if (admission(consumer, device, {device, 900, true}) != status::submitted) return false;
      value.command = 0;
      if (admission(consumer, device, {device, 900, true}) != status::ready) return false;
      value.finished = false;
      if (admission(consumer, device, {device, 900, true}) == status::ready) return false;
      value.finished = true; value.success = false;
      if (admission(consumer, device, {device, 900, true}) != status::failed) return false;
      value.success = true; value.consumer_queue = consumer;
      if (admission(other, device, {device, 900, true}) != status::unsupported_queue ||
          admission(producer, device, {}) != status::unsupported_queue) return false;

      value.command = 101; value.consumers[0] = 102; value.acquired = true;
      consumer_submitted(value, consumer, 7, true);
      consumer_submitted(value, consumer, 11, true); // Coalesce only the same timeline.
      std::uint64_t producer_complete = 500, consumer_complete = 10, other_complete = 2;
      const auto complete = [&](const fence_point &point) {
        const auto done = point.queue == producer ? producer_complete : point.queue == consumer ? consumer_complete :
          point.queue == other ? other_complete : UINT64_MAX;
        return done != UINT64_MAX && done >= point.value;
      };
      if (value.invalid || value.retirements[0].queue != consumer || value.retirements[0].value != 11 ||
          !uses_queue(value, producer) || !uses_queue(value, consumer) || uses_queue(value, other) || gpu_retired(value, complete)) return false;
      retire_recording(value, 101); retire_recording(value, 102);
      if (value.command || value.consumers[0] || value.producer_fence != 500 || value.retirements[0].value != 11 ||
          !uses_queue(value, consumer) || gpu_retired(value, complete)) return false;
      consumer_complete = 11;
      if (!gpu_retired(value, complete)) return false;
      producer_complete = 499;
      if (gpu_retired(value, complete)) return false;
      producer_complete = UINT64_MAX;
      if (gpu_retired(value, complete)) return false;
      producer_complete = 500;
      // An externally replayed consumer on a different queue is invalid, but
      // its actual fence must still pin the copy and that queue until complete.
      consumer_submitted(value, other, 3, true);
      if (!value.invalid || value.failure != capture_failure::consumer_queue_changed ||
          !uses_queue(value, other) || gpu_retired(value, complete)) return false;
      other_complete = 3;
      if (!gpu_retired(value, complete)) return false;
      value.retirement_unknown = true;
      if (gpu_retired(value, complete)) return false;
      value.retirement_unknown = false;
      consumer_submitted(value, consumer, 12, false);
      if (!value.retirement_unknown || gpu_retired(value, complete)) return false;

      slot same_queue;
      same_queue.id = 2; same_queue.queue = producer; same_queue.producer_fence = 500;
      same_queue.consumer_queue = producer;
      consumer_submitted(same_queue, producer, 501, true);
      if (same_queue.invalid || gpu_retired(same_queue, complete)) return false;
      producer_complete = 501;
      if (!gpu_retired(same_queue, complete)) return false;
      // Discarding a consumer before Execute adds no phantom GPU obligation;
      // it still cannot release an unfinished producer.
      slot discarded;
      discarded.id = 3; discarded.queue = producer; discarded.producer_fence = 502;
      discarded.producer_submitted = true; discarded.consumers[0] = 103;
      retire_recording(discarded, 103);
      if (discarded.consumers[0] || gpu_retired(discarded, complete)) return false;
      producer_complete = 502;
      return gpu_retired(discarded, complete);
    }
    bool source_authority_regression() {
      struct cleanup {
        std::array<slot, slot_limit> saved_slots{slots};
        decltype(evaluations) saved_evaluations{evaluations};
        std::array<status, provider_count> attempts{attempt_status[0].load(), attempt_status[1].load()};
        bool enabled{requested.load()}; status saved_reason{reason.load()};
        ~cleanup() {
          slots = saved_slots; evaluations = saved_evaluations; requested = enabled; reason = saved_reason;
          for (unsigned i = 0; i != provider_count; ++i) attempt_status[i] = attempts[i];
        }
      } restore;
      slots = {}; evaluations = {}; requested = true;
      auto source = std::make_shared<source_reference>();
      source->cookie = 31; source->device_identity = 41;
      source->desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
      source->desc.Width = 800; source->desc.Height = 600;
      source->desc.DepthOrArraySize = source->desc.MipLevels = source->desc.SampleDesc.Count = 1;
      source->desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM; // Valid identity/extent, unsupported depth copy.
      input metadata; metadata.source = source; metadata.epoch = 1; metadata.sequence = 1;
      metadata.source_id = 19; metadata.tick = GetTickCount64(); metadata.resource.native = 23;
      copy_region region;
      if (source_region(source->desc, metadata.resource, region) != status::ready ||
          capture_region(source->desc, metadata.resource, region) != status::unsupported_resource) return false;
      auto malformed = metadata.resource; malformed.area = {799, 0, 2, 600};
      if (source_region(source->desc, malformed, region) != status::malformed) return false;
      begin_evaluation(1, 1, 0, provider_kind::streamline, 19);
      const auto ticket = record_nomination(metadata, metadata.resource.native, 77, {});
      const auto nominated = std::find_if(slots.begin(), slots.end(), [&](const auto &value) { return value.id == ticket; });
      if (!ticket || nominated == slots.end()) return false;
      auto &value = *nominated;
      value.source_nominated = value.nomination_only = true; value.pixel_failure = status::unsupported_resource;
      queue_state owner; owner.device_identity = 41;
      constexpr std::uint64_t consumer = 101;
      const auto select = [&] { return select_provider(owner, consumer, 1, GetTickCount64()); };
      if (select().value) return false;
      finish(ticket, true);
      if (select().value) return false; // Successful but never submitted is not source authority.
      producer_submitted(value, consumer + 1, 0, true);
      auto chosen = select();
      if (chosen.value != &value || chosen.result != status::unsupported_resource ||
          nomination_status(value, owner.device_identity + 1) != status::unsupported_queue) return false;
      packet out; describe_packet(value, consumer, out);
      if (out.pixel_ready || out.texture || out.ownership || out.resource_id != 31 ||
          out.width != 800 || out.height != 600) return false;
      consumer_diagnostic consumer_result;
      if (copy_current(5, out, 7, D3D12_RESOURCE_STATE_COPY_DEST, &consumer_result) ||
          consumer_result.result != consumer_status::capture_not_ready) return false;
      establish_provider(owner, value, 1);
      if (!owner.provider_established || owner.source_id != 19) return false;
      // Native pixel submission readiness is independent too: no private fence
      // means no usable snapshot, but the successful submitted source remains.
      value.nomination_only = false; value.queue = consumer;
      if (nomination_status(value, owner.device_identity) != status::ready ||
          submission_status(value, consumer, owner.device_identity, {}) == status::ready) return false;
      value.producer_fence = 7;
      bind_pixel_consumer(value, consumer, submission_status(value, consumer, owner.device_identity, {}));
      if (value.consumer_queue != consumer) return false;
      consumer_submitted(value, consumer, 8, true);
      queue_state other_owner; other_owner.device_identity = owner.device_identity;
      // Q2 can see the same authoritative source, but Q1 already owns the
      // snapshot's transition timeline. Repeating Q2 effects cannot turn that
      // authority-only result into permission to read the same texture.
      for (unsigned attempt = 0; attempt != 2; ++attempt) {
        const auto other = select_provider(other_owner, consumer + 1, 1, GetTickCount64());
        if (other.value != &value || other.result != status::unsupported_queue) return false;
        bind_pixel_consumer(value, consumer + 1, other.result);
        establish_provider(other_owner, value, 1);
        if (value.consumer_queue != consumer ||
            submission_status(value, consumer, owner.device_identity, {}) != status::ready) return false;
      }
      value.nomination_only = true; value.success = false;
      owner.provider_established = false; owner.present_capture = 0;
      if (select().value) return false; // Failed first evaluation still permits fallback.
      value.success = true;
      invalidate(value, capture_failure::producer_signal_failed);
      if (nomination_status(value, owner.device_identity) != status::ready) return false;
      // A later source release must revoke authority even when its first pixel
      // failure remains the diagnostic reason (first-failure preservation).
      invalidate(value, capture_failure::source_retired);
      return nomination_status(value, owner.device_identity) == status::failed;
    }
    bool preservation_owner_regression() {
      struct cleanup {
        std::array<slot, slot_limit> saved_slots{slots};
        decltype(evaluations) saved_evaluations{evaluations};
        std::array<status, provider_count> attempts{attempt_status[0].load(), attempt_status[1].load()};
        status saved_reason{reason.load()};
        ~cleanup() {
          slots = saved_slots; evaluations = saved_evaluations; reason = saved_reason;
          for (unsigned i = 0; i != provider_count; ++i) attempt_status[i] = attempts[i];
        }
      } restore;
      slots = {}; evaluations = {};
      constexpr std::uint64_t device = 11, source_id = 12, producer = 17, consumer = 19, recording = 21;
      auto source = std::make_shared<source_reference>(); source->cookie = source_id; source->device_identity = device;
      auto &value = slots[0];
      value.id = 101; value.preservation_only = true; value.finished = value.success = true;
      value.command = recording; value.metadata.source = source;
      value.metadata.epoch = 3; value.metadata.sequence = 4; value.metadata.source_id = 5;
      value.metadata.tick = 100; value.texture = std::make_shared<texture_reference>();
      preservation_ticket ticket{value.id, 0, 0, value.texture};
      // A repeated before-clear opportunity can reuse an unread allocation in
      // this exact open recording; other sources/recordings and any GPU reader
      // require their own allocation. CPU ticket copies alone do not read it.
      if (!reusable_preserved(value, value.metadata, recording) ||
          reusable_preserved(value, value.metadata, recording + 1)) return false;
      value.consumers[0] = 30;
      if (reusable_preserved(value, value.metadata, recording)) return false;
      value.consumers[0] = 0;
      value.acquired = true;
      if (reusable_preserved(value, value.metadata, recording)) return false;
      value.acquired = false;
      // EOF copies can be consumed later in the same open recording. A ticket
      // on any other recording needs actual producer submission first.
      if (preserved_admission(value, ticket, consumer, device, recording) != consumer_status::ready ||
          preserved_admission(value, ticket, consumer, device, recording + 1) != consumer_status::capture_not_ready)
        return false;
      producer_submitted(value, producer, 7, true);
      if (reusable_preserved(value, value.metadata, recording) ||
          preserved_admission(value, ticket, consumer, device, recording + 1) != consumer_status::ready ||
          preserved_admission(value, ticket, consumer, device + 1, recording) != consumer_status::device_mismatch)
        return false;
      value.consumer_queue = consumer;
      if (preserved_admission(value, ticket, consumer + 1, device, recording) != consumer_status::ownership_mismatch)
        return false;
      std::array<bool, slot_limit> producers{}, consumers{}; producers[0] = true;
      if (!needs_submission_fence(producers, consumers)) return false; // Real snapshot, unlike a nomination.
      // Even matching API sequence/epoch/source metadata cannot turn a Generic
      // snapshot into API authority or make feature release destroy its lease.
      evaluations[0].epoch = 3; evaluations[0].sequence = 4; evaluations[0].source_id = 5;
      evaluations[0].views[0] = {3, 4, 100, 5, 0};
      queue_state owner; owner.device_identity = device;
      if (select_provider(owner, consumer, 1, 100).value || select_provider(owner, consumer, 1, 100).latest) return false;
      retire_source(provider_kind::streamline, 3, 5);
      if (value.invalid) return false;
      // Producer and consumer retirement still use the SAME queue-specific
      // obligations as API snapshots, not a frame-count or COM-only shortcut.
      consumer_submitted(value, consumer, 8, true);
      if (gpu_retired(value, [&](const fence_point &p) { return p.queue == producer && p.value <= 7; })) return false;
      if (!gpu_retired(value, [&](const fence_point &p) {
        return (p.queue == producer && p.value <= 7) || (p.queue == consumer && p.value <= 8);
      })) return false;
      invalidate(value, capture_failure::replay);
      if (preserved_admission(value, ticket, consumer, device, recording) != consumer_status::capture_not_ready) return false;
      // A CPU ticket keeps a completed/discarded slot out of the allocation
      // pool until the frame/command stats release it. A reused ID is rejected.
      value.invalid = false; value.producer_submitted = false; value.command = 0; value.consumers = {};
      value.retirements = {}; value.consumer_queue = 0;
      if (reclaimable(value)) return false;
      ++value.id;
      if (preserved_admission(value, ticket, consumer, device, recording) != consumer_status::ownership_mismatch) return false;
      ticket = {};
      return reclaimable(value);
    }
    bool submission_completion_regression() {
      auto available = std::find_if(slots.begin(), slots.end(), [](const auto &value) { return !value.id; });
      if (available == slots.end()) return false;
      struct cleanup {
        slot &value;
        slot saved;
        ~cleanup() { value = saved; }
      } restore{*available, *available};
      auto &value = *available;
      constexpr std::uint64_t native = 17, fence = 23;
      const auto restart = [&] { value = {}; value.id = ++serial; };
      // Drive the actual submission bookkeeping and acquire predicate, with
      // only the queue Signal result supplied by the CPU fixture.
      restart();
      producer_submitted(value, native, fence, true);
      if (value.invalid || value.finished || value.failure != capture_failure::none ||
          value.retirement_unknown || value.producer_fence != fence || submitted_success(value, native)) return false;
      finish(value.id, true);
      if (!submitted_success(value, native) || submitted_success(value, native + 1) || value.failure != capture_failure::none) return false;
      restart();
      finish(value.id, true);
      if (submitted_success(value, native)) return false;
      producer_submitted(value, native, fence, true);
      if (!submitted_success(value, native)) return false;
      // A failed result remains terminal in either ordering.
      for (const bool result_first : {false, true}) {
        restart();
        if (result_first) finish(value.id, false, capture_failure::evaluation_failed);
        producer_submitted(value, native, fence, true);
        if (!result_first) finish(value.id, false, capture_failure::evaluation_failed);
        if (!value.invalid || submitted_success(value, native) || value.failure != capture_failure::evaluation_failed) return false;
      }
      restart();
      producer_submitted(value, native, fence, false);
      finish(value.id, false);
      if (!value.invalid || !value.retirement_unknown || value.producer_fence ||
          submitted_success(value, native) || value.failure != capture_failure::producer_signal_failed) return false;
      restart();
      invalidate(value, capture_failure::observer_loss);
      finish(value.id, false);
      producer_submitted(value, native, fence, true);
      if (submitted_success(value, native) || value.failure != capture_failure::observer_loss) return false;
      restart();
      finish(value.id, true);
      producer_submitted(value, native, fence, true);
      producer_submitted(value, native + 1, fence + 1, true);
      return value.invalid && value.retirement_unknown && !submitted_success(value, native + 1) &&
        value.queue == native && value.producer_fence == fence &&
        value.failure == capture_failure::producer_queue_changed && cross_queue_completion_regression() &&
        preservation_owner_regression() && source_authority_regression();
    }
    bool snapshot_basis_regression() {
      // Provenance: admission still uses the mapped legacy state of an
      // enhanced layout, while the basis records where it came from.
      const auto saved = requested.load();
      struct restore_requested { bool value; ~restore_requested() { requested = value; } } restore{saved};
      requested = true;
      private_object list, target;
      const auto address = native_object(list);
      std::uint64_t cookie{};
      associate_recording(address, &cookie);
      auto owner = command(address, cookie, false);
      if (!cookie || !owner) return false;
      owner->reported = true;
      const native_observer::enhanced_texture shader_resource{reinterpret_cast<ID3D12Resource *>(&target),
        D3D12_BARRIER_LAYOUT_SHADER_RESOURCE, true};
      enhanced_textures(address, cookie, 1, &shader_resource);
      const auto identity = source_cookie(&target);
      const auto *entry = state(*owner, identity, false);
      if (!identity || !entry || !entry->known || entry->blocked || !entry->enhanced) return false;
      const auto chosen = copy_state::resolve(copy_state::rule::observed_else_contract, UINT32_MAX,
        copy_state::observation{true, entry->known, entry->blocked, entry->value}, 0);
      if (!chosen.admitted || !chosen.used_observed || copy_basis(entry, chosen) != state_basis::observed_enhanced) return false;
      // A later legacy transition replaces the provenance.
      D3D12_RESOURCE_BARRIER transition{};
      transition.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      transition.Transition.pResource = reinterpret_cast<ID3D12Resource *>(&target);
      transition.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
      transition.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
      barriers(address, cookie, 1, &transition);
      entry = state(*owner, identity, false);
      if (!entry || entry->enhanced || copy_basis(entry, chosen) != state_basis::observed_legacy) return false;
      copy_state::decision contract{};
      contract.used_contract = true;
      if (copy_basis(nullptr, contract) != state_basis::contract || copy_basis(nullptr, {}) != state_basis::declared) return false;
      return std::string_view(name(state_basis::observed_enhanced)) == "observed_enhanced";
    }
  }
#endif

}
