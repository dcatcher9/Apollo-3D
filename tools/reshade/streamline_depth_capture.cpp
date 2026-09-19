// SPDX-License-Identifier: GPL-3.0-only
#include "streamline_depth_capture.h"
#include "streamline_native_observer.h"
#include "native_command_storage.h"
#include "native_resource_identity.h"
#include <d3d12.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

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
    constexpr D3D12_RESOURCE_STATES sampled_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    constexpr std::uint32_t write_states = D3D12_RESOURCE_STATE_DEPTH_WRITE | D3D12_RESOURCE_STATE_COPY_DEST |
      D3D12_RESOURCE_STATE_UNORDERED_ACCESS | D3D12_RESOURCE_STATE_RENDER_TARGET;
    constexpr GUID source_guid = sunshine_native_identity::resource_guid;
    constexpr GUID queue_guid{0x091eab97, 0xfcc2, 0x47fb, {0x97, 0x3b, 0x02, 0x7c, 0x69, 0xb0, 0xd5, 0x26}};
    std::atomic<std::uint64_t> serial{1};
    std::atomic<status> reason{status::inactive};
    constexpr unsigned provider_count = 2;
    using provider_kind = sunshine_scene_depth::provider_kind;
    std::array<std::atomic<status>, provider_count> attempt_status{status::unavailable, status::unavailable};
    std::atomic<bool> requested{};
    std::atomic<preservation_available_callback> preservation_available{};
    std::mutex mutex;

    unsigned provider_index(provider_kind value) { return static_cast<unsigned>(value); }
    bool valid_provider(provider_kind value) { return provider_index(value) < provider_count; }
    void set_attempt(provider_kind provider, status value) {
      reason = value;
      if (valid_provider(provider)) attempt_status[provider_index(provider)] = value;
    }
    bool valid_projection(const sunshine_scene_depth::projection_basis &value) {
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
    bool supported_state(std::uint32_t value) {
      // Only ordinary legacy texture states. Enhanced/split/unknown states are
      // not converted into an invented legacy state for this capture.
      constexpr std::uint32_t allowed = D3D12_RESOURCE_STATE_DEPTH_WRITE | D3D12_RESOURCE_STATE_DEPTH_READ |
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
        D3D12_RESOURCE_STATE_COPY_SOURCE | D3D12_RESOURCE_STATE_COPY_DEST |
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS | D3D12_RESOURCE_STATE_RENDER_TARGET;
      if (value & ~allowed) return false;
      return !(value & write_states) || ((value & (value - 1)) == 0);
    }
    struct source_state { std::uint64_t source{}; std::uint32_t value{}; bool known{}, blocked{}; };
    struct command_state {
      std::uint64_t cookie{};
      std::uint64_t observation_generation{};
      // The number of alias/split resources in a game recording is not bounded
      // by the number of depth sources we capture. Reuse this storage on Reset.
      std::vector<source_state> states;
      bool closed{}, invalid{}, render_pass{};
      recording_loss invalidation{recording_loss::none};
      void restart(std::uint64_t next_cookie, std::uint64_t generation) noexcept {
        states.clear();
        cookie = next_cookie; observation_generation = generation;
        closed = invalid = render_pass = false;
        invalidation = recording_loss::none;
      }
    };
    using command_storage = sunshine_native_command::storage<command_state>;
    using recording_ref = std::shared_ptr<sunshine_native_command::recording_lifetime>;
    std::uint64_t command_generation{1}; // Protected by mutex, like recording payloads.
    struct queue_state {
      com_ptr<ID3D12CommandQueue> queue;
      com_ptr<ID3D12Device> device;
      com_ptr<ID3D12Fence> fence;
      std::uint64_t cookie{}, next_fence{}, last_epoch{}, last_sequence{}, nominated_epoch{}, present{}, present_capture{};
      std::uint64_t admission_after{};
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
  };
  namespace {
    struct fence_point { std::uint64_t queue{}, value{}; };
    struct slot {
      std::shared_ptr<texture_reference> texture;
      input metadata;
      std::uint64_t id{}, command{}, queue{}, producer_fence{}, consumer_queue{};
      // Values are comparable only within their own private queue timeline.
      // Reset may discard recording identities, never these GPU obligations.
      std::array<fence_point, queue_limit> retirements{};
      recording_ref producer_recording;
      std::array<std::uint64_t, 4> consumers{};
      std::array<recording_ref, 4> consumer_recordings;
      bool finished{}, success{}, acquired{}, invalid{}, producer_submitted{}, retirement_unknown{};
      bool shared_preservation{};
      bool preservation_only{};
      bool source_nominated{}, nomination_only{}, nomination_invalid{};
      status pixel_failure{status::unavailable};
      capture_failure failure{capture_failure::none};
    };
    std::array<slot, slot_limit> slots;
    std::array<std::weak_ptr<const source_reference>, source_limit> sources;
    std::array<std::unique_ptr<queue_state>, queue_limit> queues;
    struct evaluation_head { std::uint64_t epoch{}, sequence{}, tick{}, source_id{}; std::uint32_t viewport{}; };
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
      for (auto &value : slots) invalidate(value, capture_failure::observer_loss);
      ++command_generation;
      reason = status::unavailable;
    }
    void barriers(std::uint64_t native, std::uint64_t cookie, std::uint32_t count, const D3D12_RESOURCE_BARRIER *values) {
      std::lock_guard lock(mutex);
      auto owner = command(native, cookie, true);
      if (!owner || owner->closed) return;
      for (unsigned n = 0; n != count; ++n) {
        const auto &value = values[n];
        if (value.Type == D3D12_RESOURCE_BARRIER_TYPE_ALIASING) {
          // A named alias affects those resources only. A wildcard can affect
          // any source, so no later declaration can recover this recording.
          if (!value.Aliasing.pResourceBefore || !value.Aliasing.pResourceAfter) invalidate(*owner, recording_loss::wildcard_alias);
          else { block_source(*owner, value.Aliasing.pResourceBefore); block_source(*owner, value.Aliasing.pResourceAfter); }
        } else if (value.Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION) {
          // A split can precede the first tag/cookie. Preserve that source's
          // rejection until Reset without poisoning unrelated captured depth.
          if (value.Flags != D3D12_RESOURCE_BARRIER_FLAG_NONE) { block_source(*owner, value.Transition.pResource); continue; }
          const auto identity = source_cookie(value.Transition.pResource);
          if (!identity || (value.Transition.Subresource != 0 && value.Transition.Subresource != D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES)) continue;
          auto *known = state(*owner, identity, true);
          if (!known) { invalidate(*owner, recording_loss::source_state_capacity); continue; }
          if (known->blocked) continue;
          known->known = value.Flags == D3D12_RESOURCE_BARRIER_FLAG_NONE && supported_state(value.Transition.StateAfter);
          known->value = value.Transition.StateAfter;
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
    void reset(std::uint64_t native, std::uint64_t old, HRESULT result) {
      if (FAILED(result)) return;
      std::lock_guard lock(mutex);
      for (auto &value : slots) retire_recording(value, old);
      if (!native) return;
      auto owner = command_storage::acquire(reinterpret_cast<ID3D12Object *>(native), true);
      if (!owner) return;
      owner.life()->cookie.store(0, std::memory_order_release);
      const auto cookie = ++serial;
      if (!native_observer::set_recording_cookie(native, cookie)) {
        invalidate(*owner, recording_loss::source_identity_unavailable); return;
      }
      owner->restart(cookie, command_generation);
      owner.life()->cookie.store(cookie, std::memory_order_release);
    }
    void close(std::uint64_t native, std::uint64_t cookie, HRESULT result) {
      std::lock_guard lock(mutex);
      if (auto value = command(native, cookie, true)) {
        value->closed = SUCCEEDED(result);
        if (FAILED(result)) invalidate(*value, recording_loss::close_failed);
      }
      if (FAILED(result)) for (auto &value : slots) if (value.command == cookie) invalidate(value, capture_failure::close_failed);
    }
    void invalidated_command(std::uint64_t native, std::uint64_t cookie) {
      std::lock_guard lock(mutex);
      if (auto owner = command(native, cookie, true)) invalidate(*owner, recording_loss::opaque_commands);
      // Already recorded copies retain their pixels. This only denies another
      // source-state assumption later in the changed recording.
    }
    void render_pass(std::uint64_t native, std::uint64_t cookie, bool inside) {
      std::lock_guard lock(mutex);
      if (auto owner = command(native, cookie, true)) owner->render_pass = inside;
    }
    bool identify_submissions(std::array<slot, slot_limit> &captures, std::uint32_t count,
        const native_observer::command_identity *values, std::array<bool, slot_limit> &producers,
        std::array<bool, slot_limit> &consumers) {
      bool used = false;
      for (unsigned n = 0; n != count; ++n) {
        const auto cookie = values[n].recording_cookie;
        // Zero is an unobserved recording, not a producer whose recording was
        // retired by Reset. It must not change capture state or create a fence.
        if (!cookie) continue;
        for (unsigned i = 0; i != captures.size(); ++i) {
          auto &value = captures[i];
          if (!value.id) continue;
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
        value.producer_submitted && value.queue == native && (value.shared_preservation || value.nomination_only || value.producer_fence);
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
        if (!value.shared_preservation && !value.nomination_only && owner->fence.p) { out.completed = owner->fence->GetCompletedValue(); out.queried = true; }
      }
      return out;
    }
    bool producer_recording_retired(const slot &value) {
      return !value.command || (value.producer_recording &&
        value.producer_recording->cookie.load(std::memory_order_acquire) != value.command);
    }
    status submission_status(const slot &value, std::uint64_t native,
        std::uint64_t consumer_device, const queue_progress &progress) {
      if (!submitted_success(value, value.queue))
        return value.invalid || (value.finished && !value.success) ? status::failed : status::recorded;
      if (value.shared_preservation) {
        // This authorizes source metadata, not GPU access. The preservation
        // owner independently verifies this presentation's actual copy.
        return consumer_device && value.metadata.source && value.metadata.source->device_identity == consumer_device ?
          status::ready : status::unsupported_queue;
      }
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
    void bind_pixel_consumer(slot &value, std::uint64_t consumer, status pixels) {
      // Source authority may be admitted even when this queue cannot read the
      // pixels. Such a nomination must never steal an existing read timeline.
      if (pixels == status::ready && !value.shared_preservation && !value.nomination_only &&
          (!value.consumer_queue || value.consumer_queue == consumer)) value.consumer_queue = consumer;
    }
    struct capture_pick {
      slot *value{};
      const slot *latest{};
      const evaluation_head *pending_nomination{};
      status result{status::unavailable};
      queue_progress progress;
    };
    capture_pick select_latest_capture(const queue_state &owner, std::uint64_t native,
        std::uint64_t present, provider_kind provider, std::uint64_t now) {
      capture_pick out;
      const auto &current = evaluations[provider_index(provider)];
      const auto is_latest = [&](const slot &value) {
        return value.id && !value.preservation_only && value.metadata.provider == provider && value.metadata.epoch == current.epoch &&
          value.metadata.sequence == current.sequence && value.metadata.source_id == current.source_id;
      };
      for (const auto &value : slots) if (is_latest(value) && (!out.latest || value.id > out.latest->id)) out.latest = &value;
      if (out.latest && out.latest->queue != native) out.progress = producer_progress(*out.latest);
      unsigned active_views = 0;
      for (const auto &value : current.views)
        if (value.epoch == current.epoch && value.sequence && now >= value.tick && now - value.tick < sunshine_scene_depth::maximum_source_age_ms) ++active_views;
      if (active_views != 1) {
        out.result = active_views ? status::ambiguous : status::stale;
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
            if (out.result == status::ready) out.result = status::submitted;
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
    // Keep one completed real frame available while the CPU records newer FG
    // input. Completion and nomination advance independently: requiring the
    // latest nomination to finish can starve a continuously pipelined game.
    slot *completed_fg_snapshot(const input &current, std::uint64_t now) {
      if (current.provider != provider_kind::streamline || !current.frame_generation_input ||
          !current.epoch || !current.source_id) return nullptr;
      slot *best = nullptr;
      for (auto &value : slots) {
        const auto &metadata = value.metadata;
        if (!value.id || !value.texture || value.shared_preservation || value.nomination_only || value.preservation_only ||
            !value.source_nominated || !value.finished || !value.success || value.invalid || value.nomination_invalid ||
            !value.producer_submitted || !producer_recording_retired(value) ||
            metadata.provider != current.provider || !metadata.frame_generation_input ||
            metadata.epoch != current.epoch || metadata.source_id != current.source_id || metadata.viewport != current.viewport ||
            metadata.observation_revision != current.observation_revision || metadata.sequence > current.sequence ||
            !metadata.tick || now < metadata.tick || now - metadata.tick >= sunshine_scene_depth::maximum_source_age_ms) continue;
        const auto progress = producer_progress(value);
        if (!progress.valid() || progress.completed < value.producer_fence) continue;
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
    capture_pick select_capture(const queue_state &owner, std::uint64_t native,
        std::uint64_t present, provider_kind provider, std::uint64_t now) {
      auto chosen = select_latest_capture(owner, native, present, provider, now);
      const auto *latest = chosen.latest;
      // Missing/failed/ambiguous nominations still revoke depth. Only a known
      // pending snapshot from the established FG source permits older pixels.
      if (!latest || (chosen.result != status::recorded && chosen.result != status::submitted) ||
          latest->invalid || latest->nomination_invalid || latest->nomination_only ||
          (latest->finished && !latest->success) || !owner.provider_established || owner.provider != provider ||
          owner.nominated_epoch != latest->metadata.epoch || owner.source_id != latest->metadata.source_id ||
          owner.viewport != latest->metadata.viewport) return chosen;
      auto *completed = completed_fg_snapshot(latest->metadata, now);
      if (!completed || completed->id <= owner.admission_after || completed->metadata.sequence >= latest->metadata.sequence ||
          (owner.last_epoch == completed->metadata.epoch && completed->metadata.sequence <= owner.last_sequence) ||
          (owner.present == present && owner.present_capture && owner.present_capture != completed->id) ||
          !same_depth_layout(completed->metadata, latest->metadata)) return chosen;
      const auto progress = producer_progress(*completed);
      if (submission_status(*completed, native, owner.device_identity, progress) != status::ready) return chosen;
      chosen.value = completed;
      chosen.progress = progress;
      chosen.result = status::ready;
      return chosen;
    }
    capture_pick select_provider(const queue_state &owner, std::uint64_t native,
        std::uint64_t present, std::uint64_t now, selection_policy policy = {}) {
      if (policy.require_frame_generation) {
        if (!policy.epoch) return {};
        // FG owns the source choice before pixels are ready. Do not silently
        // substitute SR depth while its matching camera belongs to SL FG.
        auto fg = select_capture(owner, native, present, provider_kind::streamline, now);
        const auto matches = [&](const sunshine_scene_depth::frame &value) {
          return value.provider == provider_kind::streamline && value.frame_generation_input &&
            value.epoch == policy.epoch && value.viewport == policy.viewport &&
            value.source_id == ((1ull << 63) | policy.viewport);
        };
        if ((fg.value && !matches(fg.value->metadata)) || (fg.latest && !matches(fg.latest->metadata)) ||
            (fg.pending_nomination && (fg.pending_nomination->epoch != policy.epoch ||
              fg.pending_nomination->viewport != policy.viewport ||
              fg.pending_nomination->source_id != ((1ull << 63) | policy.viewport)))) return {};
        return fg;
      }
      if (owner.provider_established) return select_capture(owner, native, present, owner.provider, now);
      capture_pick chosen;
      for (unsigned index = 0; index != provider_count; ++index) {
        auto candidate = select_capture(owner, native, present, static_cast<provider_kind>(index), now);
        if (candidate.value) {
          if (!chosen.value || candidate.value->id < chosen.value->id) chosen = candidate;
        } else if (!chosen.value && (candidate.latest ? !chosen.latest || candidate.latest->id > chosen.latest->id :
            !chosen.latest && evaluations[index].sequence)) chosen = candidate;
      }
      return chosen;
    }
    void establish_provider(queue_state &owner, const slot &value, std::uint64_t present) {
      if (!owner.provider_established || owner.provider != value.metadata.provider ||
          owner.source_id != value.metadata.source_id || owner.nominated_epoch != value.metadata.epoch)
        owner.last_epoch = owner.last_sequence = 0;
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
        if ((!slots[i].shared_preservation && !slots[i].nomination_only && producers[i]) || consumers[i]) return true;
      return false;
    }
    void submitted(std::uint64_t native, std::uint32_t count, const native_observer::command_identity *values) {
      std::lock_guard lock(mutex);
      auto *owner = queue(native);
      if (!owner) return;
      native = reinterpret_cast<std::uint64_t>(owner->queue.p);
      std::array<bool, slot_limit> producers{}, consumers{};
      if (!identify_submissions(slots, count, values, producers, consumers)) return;
      const bool needs_fence = needs_submission_fence(producers, consumers);
      const auto fence = needs_fence ? ++owner->next_fence : 0;
      // This callback runs AFTER the actual native ExecuteCommandLists. Signal
      // therefore follows the copied/consumed pixels on this queue.
      const bool signaled = !needs_fence || SUCCEEDED(owner->queue->Signal(owner->fence.p, fence));
      for (unsigned i = 0; i != slots.size(); ++i) {
        auto &value = slots[i];
        if (producers[i]) producer_submitted(value, native, value.shared_preservation || value.nomination_only ? 0 : fence,
          value.shared_preservation || value.nomination_only || signaled);
        if (consumers[i]) {
          consumer_submitted(value, native, fence, signaled);
          // Keep the recording reference until Reset/destroy. A legal replay of
          // an unchanged command list can consume this texture again.
        }
      }
    }
    bool reclaimable(slot &value) {
      retire_dead_recordings(value);
      if (!value.id) return true;
      if (value.shared_preservation || value.nomination_only) {
        // No GPU commands/resources belong to a nomination. Superseded or
        // rejected metadata can be discarded even if the game's list remains
        // closed. Retain the latest pending/successful ticket for acquisition.
        const auto &head = evaluations[provider_index(value.metadata.provider)];
        return value.invalid || (value.finished && !value.success) ||
          value.metadata.epoch != head.epoch || value.metadata.sequence != head.sequence ||
          value.metadata.source_id != head.source_id;
      }
      if (value.retirement_unknown) return false;
      if (value.command) return false; // A closed producer may legally replay.
      if (value.texture.use_count() > 1) return false;
      if (std::any_of(value.consumers.begin(), value.consumers.end(), [](auto v) { return v != 0; })) return false;
      if (!value.producer_submitted) return value.finished && value.command == 0;
      return gpu_retired(value, [](const fence_point &point) {
        const auto *owner = known_queue(point.queue);
        if (!owner || !owner->fence.p) return false;
        const auto complete = owner->fence->GetCompletedValue();
        return complete != UINT64_MAX && complete >= point.value;
      });
    }
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
      found->shared_preservation = true;
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
      const auto desc = value.shared_preservation || value.nomination_only ? value.metadata.source->desc : value.texture->resource->GetDesc();
      out.metadata = value.metadata; out.ownership = value.texture;
      out.shared_preservation = value.shared_preservation;
      out.resource_id = value.metadata.source ? value.metadata.source->cookie : 0;
      if (value.shared_preservation || value.nomination_only) out.device = reinterpret_cast<std::uint64_t>(value.metadata.source->device.p);
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
      out.pixel_ready = !value.shared_preservation && !value.nomination_only;
    }
    void collect_retired() {
      for (auto &owner : queues) {
        if (!owner || !owner->retiring) continue;
        const auto native = reinterpret_cast<std::uint64_t>(owner->queue.p);
        const bool removed = FAILED(owner->device->GetDeviceRemovedReason());
        bool retained = false;
        for (auto &value : slots) if (value.id && uses_queue(value, native)) {
          if (removed || reclaimable(value)) value = {};
          else retained = true;
        }
        if (!retained) owner.reset();
      }
    }
  }

  void initialize(bool enabled) {
    requested = false;
    {
      std::lock_guard lock(mutex);
      evaluations = {};
      for (auto &attempt : attempt_status) attempt = status::unavailable;
      for (auto &owner : queues) if (owner) {
        owner->provider_established = false;
        owner->source_id = owner->last_epoch = owner->last_sequence = owner->nominated_epoch = owner->present_capture = 0;
        owner->admission_after = 0;
      }
    }
    requested = enabled;
    reason = enabled ? status::unavailable : status::inactive;
    native_observer::callbacks callbacks;
    callbacks.barriers = barriers; callbacks.reset = reset; callbacks.close = close;
    callbacks.submitted = submitted; callbacks.invalidated = invalidate_all;
    callbacks.invalidated_command = invalidated_command; callbacks.render_pass = render_pass;
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
    if (!native_observer::get_recording_cookie(native)) native_observer::set_recording_cookie(native, ++serial);
    {
      std::lock_guard lock(mutex);
      // Attach before hook discovery can invalidate the current recording.
      // Late payload allocation must not erase a loss already observed here.
      command(native, native_observer::get_recording_cookie(native), true);
    }
    native_observer::observe_command(native);
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
    for (auto &value : slots) retire_recording(value, cookie);
    if (auto owner = command_storage::acquire(checked.p, false)) {
      owner.life()->cookie.store(0, std::memory_order_release);
      owner->closed = true;
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
      for (auto &value : slots) if (uses_queue(value, native)) invalidate(value, capture_failure::queue_retired);
    }
    collect_retired();
  }
  void poll() {
    if (requested.load()) native_observer::install_pending();
    std::lock_guard lock(mutex);
    collect_retired();
  }
  void set_preservation_available(preservation_available_callback callback) {
    preservation_available.store(callback, std::memory_order_release);
  }

  void observe_provider(std::uint64_t native) {
    // API activity is discovery evidence only. The actual submitted capture's
    // queue is unknown here, and depth/metadata/evaluation may still be invalid.
    observe_command(native);
  }
  bool provider_active(std::uint64_t native) {
    if (!requested.load() || !native) return false;
    std::lock_guard lock(mutex);
    const auto *owner = queue(native);
    return requested.load() && owner && !owner->retiring && owner->provider_established;
  }
  bool provider_identity(std::uint64_t native, provider_kind &provider, std::uint64_t &source_id) {
    source_id = 0;
    if (!requested.load() || !native) return false;
    std::lock_guard lock(mutex);
    const auto *owner = queue(native);
    if (!owner || owner->retiring || !owner->provider_established) return false;
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
    for (auto &value : slots) if (value.id && !value.preservation_only && matches(value.metadata)) invalidate(value, capture_failure::source_retired);
    for (auto &owner : queues) if (owner && owner->provider_established && owner->provider == provider &&
        owner->nominated_epoch == epoch && owner->source_id == source_id) {
      owner->provider_established = false;
      owner->source_id = owner->last_epoch = owner->last_sequence = owner->nominated_epoch = owner->present_capture = 0;
      // The next provider must submit a new frame after release. An otherwise
      // recent snapshot captured while the old provider owned presentation is
      // historical, and must not be resurrected by an FG Off/On transition.
      // IDs use pre-increment: serial is the last issued ID, not the next one.
      owner->admission_after = serial.load(std::memory_order_relaxed);
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

  static bool source_lifetime_current(const input &value, std::uint64_t current_generation) {
    // OnlyValidNow is captured synchronously inside the authenticated tag call.
    // An independent FG Present does not end that API-owned lifetime. This does
    // not extend UntilPresent tags or authorize a delayed preservation copy.
    if (value.valid_until == sunshine_scene_depth::lifetime::at_call && value.force_snapshot) return true;
    return value.source_present_generation && value.source_present_generation == current_generation;
  }

  static std::uint64_t record_impl(std::uint64_t native, const input &value, record_diagnostic *diagnostic,
      bool preservation_only, preservation_ticket *preserved = nullptr, bool nominate_source = false) {
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
            entry.shared_preservation = false; entry.nomination_only = entry.source_nominated = true;
            entry.pixel_failure = result; break;
          }
        }
        return ticket;
      }
      return 0;
    };
    if (!requested.load()) return reject(status::malformed, record_stage::inactive);
    if (!native) return reject(status::malformed, record_stage::malformed_input);
    if (!value.source) return reject(status::malformed, record_stage::missing_source);
    if ((!preservation_only && (!valid_provider(value.provider) || !valid_projection(value.projection) ||
        !value.epoch || !value.sequence)) || !value.resource.native)
      return reject(status::malformed, record_stage::malformed_input);
    if (value.valid_until != sunshine_scene_depth::lifetime::until_present &&
        value.valid_until != sunshine_scene_depth::lifetime::until_evaluation &&
        !(value.valid_until == sunshine_scene_depth::lifetime::at_call && value.force_snapshot))
      return reject(status::unsupported_lifetime, record_stage::unsupported_lifetime);
    const auto preservation_lookup = preservation_only || value.force_snapshot ? nullptr : preservation_available.load(std::memory_order_acquire);
    if (!preservation_lookup && value.proof != sunshine_scene_depth::state_proof::declared &&
        value.proof != sunshine_scene_depth::state_proof::observed_nonzero)
      return reject(status::unsupported_state, record_stage::unsupported_proof);
    com_ptr<ID3D12GraphicsCommandList> checked;
    if (!query_native(native, IID_ID3D12GraphicsCommandList, checked))
      return reject(status::unsupported_queue, record_stage::command_interface);
    com_ptr<ID3D12Resource> checked_source;
    if (!query_native(value.resource.native, IID_ID3D12Resource, checked_source) ||
        source_cookie(checked_source.p) != value.source->cookie)
      return reject(status::unsupported_resource, record_stage::source_identity);
    // Never hold our capture lock while the preservation registry takes its
    // resource-map lock. The retained source keeps this exact object alive.
    bool shared_preservation = false;
    if (preservation_lookup) {
      try { shared_preservation = preservation_lookup(reinterpret_cast<std::uint64_t>(checked_source.p),
        value.source->cookie, value.source->device_identity); }
      catch (...) { shared_preservation = false; }
    }
    if (!shared_preservation && value.proof != sunshine_scene_depth::state_proof::declared &&
        value.proof != sunshine_scene_depth::state_proof::observed_nonzero)
      return reject(status::unsupported_state, record_stage::unsupported_proof);
    native = reinterpret_cast<std::uint64_t>(checked.p);
    if (diagnostic) diagnostic->command = native;
    observe_command(native);
    if (!native_observer::command_ready(native)) return reject(status::unavailable, record_stage::observer_coverage);
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
    std::lock_guard lock(mutex);
    const auto current_generation = source_present_generation(value.source);
    if (diagnostic) diagnostic->current_generation = current_generation;
    if (!source_lifetime_current(value, current_generation))
      return reject(status::unsupported_lifetime, record_stage::source_lifetime);
    auto owner = command(native, cookie, true);
    if (!owner) return reject(status::unavailable, record_stage::recording_missing);
    if (diagnostic) {
      diagnostic->loss = owner->invalidation; diagnostic->recording_closed = owner->closed;
      diagnostic->recording_invalid = owner->invalid; diagnostic->render_pass = owner->render_pass;
    }
    if (owner->closed) return reject(status::unavailable, record_stage::recording_closed);
    const auto &desc = value.source->desc;
    copy_region region;
    const auto source_status = source_region(desc, value.resource, region);
    if (source_status != status::ready) return reject(source_status, record_stage::resource_region);
    valid_nomination = nominate_source;
    nomination_command = cookie; nomination_source = reinterpret_cast<std::uint64_t>(checked_source.p);
    nomination_recording = owner.life();
    const auto region_status = capture_region(desc, value.resource, region);
    if (region_status != status::ready) return reject(region_status, record_stage::resource_region);
    if (shared_preservation) {
      const auto ticket = record_nomination(value, reinterpret_cast<std::uint64_t>(checked_source.p), cookie, owner.life());
      if (!ticket) return reject(status::exhausted, record_stage::capacity);
      if (nominate_source) for (auto &entry : slots) if (entry.id == ticket) { entry.source_nominated = true; break; }
      // This ticket records only which source the successful API evaluation
      // actually submitted. State tracking belongs to the transport doing the
      // GPU copy, so missing/blocked depth state cannot reject this nomination.
      set_attempt(value.provider, status::recorded);
      if (diagnostic) { diagnostic->result = status::recorded; diagnostic->stage = record_stage::recorded; }
      return ticket;
    }
    if (owner->invalid) return reject(status::unavailable, record_stage::recording_invalid);
    if (owner->render_pass) return reject(status::unavailable, record_stage::recording_render_pass);
    auto before = value.native_state;
    const auto *observed = state(*owner, value.source->cookie, false);
    if (diagnostic && observed) {
      diagnostic->observed = true; diagnostic->observed_state = observed->value; diagnostic->blocked = observed->blocked;
    }
    // An explicit/provider state cannot override an in-progress split barrier
    // or another transition that we know cannot be represented safely.
    if (observed && (observed->blocked || !observed->known)) return reject(status::incomplete_state, record_stage::incomplete_state);
    if (value.proof == sunshine_scene_depth::state_proof::observed_nonzero) {
      if (!observed || observed->value == 0) return reject(status::missing_state, record_stage::missing_state);
      before = observed->value;
    } else if (observed && observed->value != before) {
      return reject(status::conflicting_state, record_stage::conflicting_state);
    }
    if (!supported_state(before)) return reject(status::unsupported_state, record_stage::unsupported_state);
    // Multiple legal clear boundaries in one unsubmitted recording may replace
    // the same source snapshot before anyone reads it. Old tickets are retired
    // by ID; their CPU leases alone do not represent recorded GPU reads.
    const auto pixel_end = slots.begin() + pixel_slot_limit;
    auto found = preservation_only ? std::find_if(slots.begin(), pixel_end, [&](const auto &entry) {
      return reusable_preserved(entry, value, cookie);
    }) : pixel_end;
    // Do not overwrite the only newly completed FG snapshot with its pending
    // successor before the effects queue can consume it. This borrows an
    // existing pool slot; no extra owner, queue, or resource copy is introduced.
    const auto *completed = nominate_source ? completed_fg_snapshot(value, GetTickCount64()) : nullptr;
    if (found == pixel_end) found = std::find_if(slots.begin(), pixel_end, [&](auto &entry) {
      return &entry != completed && reclaimable(entry);
    });
    if (found == pixel_end) return reject(status::exhausted, record_stage::capacity);
    auto texture = found->texture;
    const auto texture_format = typeless(desc.Format);
    if (!texture || texture->device_identity != value.source->device_identity || texture->resource->GetDesc().Width != region.width ||
        texture->resource->GetDesc().Height != region.height || texture->resource->GetDesc().Format != texture_format) {
      texture = std::make_shared<texture_reference>();
      value.source->device->AddRef(); texture->device.p = value.source->device.p;
      texture->device_identity = value.source->device_identity;
      auto destination = desc; destination.Format = texture_format; destination.Flags = D3D12_RESOURCE_FLAG_NONE;
      destination.Width = region.width; destination.Height = region.height;
      D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
      native_observer::suppression_scope suppress;
      if (FAILED(texture->device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &destination, sampled_state,
          nullptr, __uuidof(ID3D12Resource), reinterpret_cast<void **>(texture->resource.put()))))
        return reject(status::failed, record_stage::texture_allocation);
      texture->identity = retain_source_cookie(texture->resource.p);
      if (!texture->identity) return reject(status::failed, record_stage::texture_allocation);
      D3D12_DESCRIPTOR_HEAP_DESC descriptors{};
      descriptors.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; descriptors.NumDescriptors = 1;
      if (FAILED(texture->device->CreateDescriptorHeap(&descriptors, __uuidof(ID3D12DescriptorHeap),
          reinterpret_cast<void **>(texture->views.put())))) return reject(status::failed, record_stage::descriptor_allocation);
      const auto handle = texture->views->GetCPUDescriptorHandleForHeapStart();
      D3D12_SHADER_RESOURCE_VIEW_DESC view{};
      view.Format = view_format(desc.Format); view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
      view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; view.Texture2D.MipLevels = 1;
      texture->device->CreateShaderResourceView(texture->resource.p, &view, handle);
      texture->shader_resource = handle.ptr;
    }
    *found = {};
    found->texture = std::move(texture); found->metadata = value; found->metadata.native_state = before;
    found->id = ++serial; found->command = cookie; found->producer_recording = owner.life();
    found->preservation_only = preservation_only;
    found->source_nominated = nominate_source;
    found->finished = found->success = preservation_only;
    native_observer::suppression_scope suppress;
    D3D12_RESOURCE_BARRIER transitions[2]{};
    transitions[0].Type = transitions[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    transitions[0].Transition = {value.source->resource.p, 0, static_cast<D3D12_RESOURCE_STATES>(before), D3D12_RESOURCE_STATE_COPY_SOURCE};
    transitions[1].Transition = {found->texture->resource.p, 0, sampled_state, D3D12_RESOURCE_STATE_COPY_DEST};
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
  std::uint64_t record(std::uint64_t command, const input &value, record_diagnostic *diagnostic) {
    return record_impl(command, value, diagnostic, false);
  }
  std::uint64_t nominate(std::uint64_t command, const input &value, record_diagnostic *diagnostic) {
    auto source = value;
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
    if (!candidates || !count || count > 2) return malformed();
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
    out = {};
    if (diagnostic) {
      *diagnostic = {}; diagnostic->requested_queue = native;
      if (policy.require_frame_generation) {
        diagnostic->provider = provider_kind::streamline;
        diagnostic->epoch = policy.epoch; diagnostic->viewport = policy.viewport;
        diagnostic->source_id = (1ull << 63) | policy.viewport;
      }
    }
    queue_progress progress;
    const auto result = [diagnostic, &progress](status why, const slot *value = nullptr) {
      if (diagnostic) {
        diagnostic->result = why;
        if (value) {
          diagnostic->failure = value->failure;
          diagnostic->capture_id = value->id; diagnostic->sequence = value->metadata.sequence;
          diagnostic->epoch = value->metadata.epoch; diagnostic->command = value->command;
          diagnostic->source_id = value->metadata.source_id;
          diagnostic->viewport = value->metadata.viewport;
          diagnostic->provider = value->metadata.provider;
          diagnostic->queue = value->queue; diagnostic->producer_fence = value->producer_fence;
          diagnostic->producer_completed = progress.completed;
          diagnostic->producer_completion_valid = progress.valid();
          diagnostic->producer_recording_retired = producer_recording_retired(*value);
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
    const auto chosen = select_provider(*owner, native, present, GetTickCount64(), policy);
    progress = chosen.progress;
    // A genuine source interruption revokes snapshots from before the gap.
    // Normal pending/repeated frames do not move this admission watermark.
    if (owner->provider_established && owner->provider == provider_kind::streamline && owner->source_id &&
        chosen.result != status::ready && chosen.result != status::recorded && chosen.result != status::submitted)
      owner->admission_after = serial.load(std::memory_order_relaxed);
    if (diagnostic) diagnostic->newest_sequence = chosen.latest ? chosen.latest->metadata.sequence :
      chosen.pending_nomination ? chosen.pending_nomination->sequence : 0;
    if (diagnostic && chosen.pending_nomination) {
      const auto &pending = *chosen.pending_nomination;
      diagnostic->pending_frame = true;
      diagnostic->epoch = pending.epoch; diagnostic->sequence = pending.sequence;
      diagnostic->source_id = pending.source_id; diagnostic->viewport = pending.viewport;
      diagnostic->provider = policy.require_frame_generation ? provider_kind::streamline : owner->provider;
    }
    if (diagnostic) diagnostic->pending_frame = pending_frame(chosen, *owner);
    if (diagnostic) {
      const auto *value = chosen.value ? chosen.value : chosen.latest;
      if (value && value->queue == native) progress = producer_progress(*value);
    }
    if (!chosen.value) {
      if (diagnostic && chosen.latest && chosen.result != status::failed && chosen.result != status::stale &&
          chosen.result != status::ambiguous && chosen.latest->finished && chosen.latest->success &&
          !chosen.latest->nomination_invalid && !chosen.latest->invalid &&
          owner->last_epoch == chosen.latest->metadata.epoch && owner->last_sequence == chosen.latest->metadata.sequence)
        diagnostic->repeated_frame = true;
      return result(chosen.result, chosen.latest);
    }
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
  }
  static bool consume_owned(std::uint64_t native, const packet &value, std::uint64_t destination,
      std::uint32_t destination_state, consumer_diagnostic *diagnostic) {
    if (diagnostic) *diagnostic = {};
    const auto result = [diagnostic](consumer_status why) {
      if (diagnostic) diagnostic->result = why;
      return why == consumer_status::ready;
    };
    if (!native || !value.ownership) return result(consumer_status::missing_input);
    com_ptr<ID3D12GraphicsCommandList> checked;
    if (!query_native(native, IID_ID3D12GraphicsCommandList, checked)) return result(consumer_status::unsupported_interface);
    native = reinterpret_cast<std::uint64_t>(checked.p);
    observe_command(native);
    if (!native_observer::command_ready(native)) return result(consumer_status::observer_not_ready);
    const auto cookie = native_observer::get_recording_cookie(native);
    if (diagnostic) diagnostic->cookie = cookie;
    if (!cookie) return result(consumer_status::missing_cookie);
    auto *list = checked.p;
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
    if (destination) {
      if (!query_native(destination, IID_ID3D12Resource, target) || target.p == value.ownership->resource.p ||
          !supported_state(destination_state)) return result(consumer_status::invalid_destination);
      com_ptr<ID3D12Device> target_device;
      if (FAILED(target->GetDevice(IID_ID3D12Device, reinterpret_cast<void **>(target_device.put()))) ||
          device_cookie(target_device.p) != device_identity) return result(consumer_status::device_mismatch);
      const auto source_desc = value.ownership->resource->GetDesc(), target_desc = target->GetDesc();
      if (!supported_description(target_desc) || target_desc.Width != source_desc.Width || target_desc.Height != source_desc.Height ||
          typeless(target_desc.Format) != typeless(source_desc.Format)) return result(consumer_status::invalid_destination);
    }
    std::lock_guard lock(mutex);
    auto recording = command(native, cookie, true);
    if (!recording) {
      if (diagnostic) diagnostic->tracked_commands = command_storage::live_count();
      return result(consumer_status::recording_missing);
    }
    if (diagnostic) diagnostic->invalidation = recording->invalidation;
    if (recording->closed) return result(consumer_status::recording_closed);
    if (recording->invalid) return result(consumer_status::recording_invalid);
    if (recording->render_pass) return result(consumer_status::recording_render_pass);
    bool matching_id = false;
    for (auto &entry : slots) if (entry.id == value.capture_id) {
      matching_id = true;
      if (entry.texture != value.ownership) continue;
      if (entry.consumer_queue != value.queue) return result(consumer_status::ownership_mismatch);
      retire_dead_recordings(entry);
      if (diagnostic) diagnostic->slot_invalid = entry.invalid;
      if (entry.invalid || !entry.finished || !entry.success) return result(consumer_status::capture_not_ready);
      for (unsigned i = 0; i != entry.consumers.size(); ++i) if (!entry.consumers[i] || entry.consumers[i] == cookie) {
        if (!entry.preservation_only && entry.queue != value.queue) {
          // Recheck the actual slot before recording any read. A pending copy
          // must never add a queue dependency to the application's schedule.
          const auto *consumer = known_queue(value.queue);
          const auto progress = producer_progress(entry);
          if (!consumer || consumer->retiring || !producer_recording_retired(entry) ||
              !progress.valid() || !entry.producer_fence || progress.completed < entry.producer_fence)
            return result(consumer_status::capture_not_ready);
        }
        entry.consumers[i] = cookie; entry.consumer_recordings[i] = recording.life();
        entry.acquired = true;
        if (target.p) {
          // All selected sources use this exact copy and consumer lease path.
          // Full depth plane only: stencil and packed padding are untouched.
          native_observer::suppression_scope suppress;
          D3D12_RESOURCE_BARRIER transitions[2]{};
          transitions[0].Type = transitions[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
          transitions[0].Transition = {entry.texture->resource.p, 0, sampled_state, D3D12_RESOURCE_STATE_COPY_SOURCE};
          transitions[1].Transition = {target.p, 0, static_cast<D3D12_RESOURCE_STATES>(destination_state), D3D12_RESOURCE_STATE_COPY_DEST};
          list->ResourceBarrier(destination_state == D3D12_RESOURCE_STATE_COPY_DEST ? 1 : 2, transitions);
          D3D12_TEXTURE_COPY_LOCATION source_location{}, target_location{};
          source_location.pResource = entry.texture->resource.p; source_location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
          target_location.pResource = target.p; target_location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
          list->CopyTextureRegion(&target_location, 0, 0, 0, &source_location, nullptr);
          std::swap(transitions[0].Transition.StateBefore, transitions[0].Transition.StateAfter);
          std::swap(transitions[1].Transition.StateBefore, transitions[1].Transition.StateAfter);
          list->ResourceBarrier(destination_state == D3D12_RESOURCE_STATE_COPY_DEST ? 1 : 2, transitions);
        }
        return result(consumer_status::ready);
      }
      invalidate(entry, capture_failure::consumer_capacity); reason = status::exhausted; return result(consumer_status::consumer_capacity);
    }
    return result(matching_id ? consumer_status::ownership_mismatch : consumer_status::slot_missing);
  }
  bool mark_consumer(std::uint64_t command, const packet &value, consumer_diagnostic *diagnostic) {
    if (!value.pixel_ready) {
      if (diagnostic) { *diagnostic = {}; diagnostic->result = consumer_status::capture_not_ready; }
      return false;
    }
    return consume_owned(command, value, 0, 0, diagnostic);
  }
  bool copy_current(std::uint64_t command, const packet &value, std::uint64_t destination,
      std::uint32_t destination_state, consumer_diagnostic *diagnostic) {
    if (!destination || !value.pixel_ready) {
      if (diagnostic) { *diagnostic = {}; diagnostic->result = !destination ? consumer_status::invalid_destination : consumer_status::capture_not_ready; }
      return false;
    }
    return consume_owned(command, value, destination, destination_state, diagnostic);
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
    observe_command(command);
    observe_queue(consumer_queue);
    const auto cookie = native_observer::get_recording_cookie(command);
    packet selected;
    {
      std::lock_guard lock(mutex);
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
    CASE(evaluation_failed); CASE(evaluation_observation_changed); CASE(queue_retired); CASE(consumer_capacity); CASE(source_retired);
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
    CASE(none); CASE(global_observation_loss); CASE(wildcard_alias); CASE(source_identity_unavailable);
    CASE(source_state_capacity); CASE(close_failed); CASE(opaque_commands);
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
      return valid_projection(projection);
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
      queue_state owner;
      constexpr std::uint64_t native = 77;
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
      if (owner.provider_established || select(1).value) return false;
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
      owner.last_epoch = ngx.metadata.epoch; owner.last_sequence = ngx.metadata.sequence;
      // A newer SL success cannot replace the established NGX source, and an
      // invalid SL attempt cannot revoke its reusable current presentation.
      make(1, provider_kind::streamline, 1, 101, 0);
      if (select(1).value != &ngx || select(2).value) return false;
      slots[1].metadata.frame_generation_input = true;
      slots[1].metadata.source_id = 1ull << 63;
      slots[1].metadata.sequence = 102;
      begin_evaluation(1, 102, 0, provider_kind::streamline, slots[1].metadata.source_id, 0);
      if (select(2).value || select_provider(owner, native, 2, GetTickCount64(), {true, 1, 0}).value != &slots[1]) return false;
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
      // Fresh queues arbitrate by accepted capture order, never module order.
      slots = {}; evaluations = {};
      owner.provider_established = false; owner.present_capture = owner.last_sequence = owner.last_epoch = 0;
      auto &first = make(0, provider_kind::ngx, 3, 1, 123);
      make(1, provider_kind::streamline, 3, 1, 0);
      if (select(4).value != &first) return false;
      // Simultaneous distinct view/source identities remain ambiguous.
      begin_evaluation(3, 2, 1, provider_kind::ngx, 124);
      const auto ambiguous = select_capture(owner, native, 4, provider_kind::ngx, GetTickCount64());
      if (ambiguous.value || ambiguous.result != status::ambiguous) return false;
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
      input alternatives[]{candidate, candidate};
      if (nominate_evaluation(0, alternatives, 2, UINT64_MAX, &rejected) || rejected.stage != record_stage::malformed_input ||
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
          nominate_evaluation(0, alternatives, 3, UINT64_MAX, &rejected) || !pending_pick().pending_nomination) return false;
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
    namespace {
      struct private_object final : ID3D12Object {
        ULONG references{1};
        std::uint64_t cookie{};
        unsigned writes{};
        bool refuse_write{};
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
      private_object command_object, target, alias_peer;
      std::array<private_object, 160> unrelated;
      const auto native_command = native_object(command_object);
      const auto cookie = ++serial;
      auto recording = command(native_command, cookie, true);
      if (!recording) return false;
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
      D3D12_RESOURCE_BARRIER alias{};
      alias.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
      alias.Aliasing.pResourceBefore = reinterpret_cast<ID3D12Resource *>(&target);
      alias.Aliasing.pResourceAfter = reinterpret_cast<ID3D12Resource *>(&alias_peer);
      barriers(native_command, cookie, 1, &alias);
      if (!append_unrelated(80, unsigned(unrelated.size()))) return false;
      barriers(native_command, cookie, 1, &transition);
      selected = state(*recording, target_id, false);
      if (recording->invalid || !selected || !selected->blocked || recording->states.size() != 162) return false;
      const auto capacity = recording->states.capacity();
      const auto *storage = recording->states.data();
      reset(native_command, cookie, S_OK);
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
        recording.life()->cookie.store(cookie);
      };
      *capture = {}; // Match record() allocation after a prior global invalidation.
      capture->id = ++serial; capture->command = cookie; capture->finished = capture->success = capture->producer_submitted = true;
      D3D12_RESOURCE_BARRIER value{};
      value.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      // All calls made here use ID3D12Object's inherited private-data methods.
      value.Transition.pResource = reinterpret_cast<ID3D12Resource *>(&native);
      value.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      barriers(native_command, cookie, 1, &value);
      if (recording->invalid || native.cookie) return false; // Ordinary untagged resources are not registered.
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
      value = {}; value.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
      value.Aliasing.pResourceBefore = reinterpret_cast<ID3D12Resource *>(&unrelated);
      value.Aliasing.pResourceAfter = reinterpret_cast<ID3D12Resource *>(&alias_peer);
      barriers(native_command, cookie, 1, &value);
      const auto *unrelated_state = state(*recording, unrelated.cookie, false);
      const auto *alias_state = state(*recording, alias_peer.cookie, false);
      selected_state = state(*recording, selected, false);
      if (recording->invalid || selected_state->blocked || !selected_state->known || !unrelated_state ||
          !unrelated_state->blocked || !alias_state || !alias_state->blocked || capture->invalid) return false;
      // A named alias touching selected depth blocks it even after a normal transition.
      value.Aliasing.pResourceBefore = reinterpret_cast<ID3D12Resource *>(&native);
      barriers(native_command, cookie, 1, &value);
      value = {}; value.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      value.Transition.pResource = reinterpret_cast<ID3D12Resource *>(&native);
      value.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      barriers(native_command, cookie, 1, &value);
      if (recording->invalid || !selected_state->blocked || capture->invalid) return false;
      // Without exact affected identities or sufficient memory, decline all.
      for (unsigned wildcard = 0; wildcard != 3; ++wildcard) {
        restart_case();
        value = {}; value.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
        if (wildcard == 1) value.Aliasing.pResourceBefore = reinterpret_cast<ID3D12Resource *>(&native);
        if (wildcard == 2) value.Aliasing.pResourceAfter = reinterpret_cast<ID3D12Resource *>(&native);
        barriers(native_command, cookie, 1, &value);
        if (!recording->invalid || capture->invalid) return false;
      }
      restart_case();
      refused.refuse_write = true;
      value = {}; value.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      value.Flags = D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY;
      value.Transition.pResource = reinterpret_cast<ID3D12Resource *>(&refused);
      barriers(native_command, cookie, 1, &value);
      if (!recording->invalid || capture->invalid) return false;
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
      reset(native_command, cookie, S_OK);
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
    bool shared_preservation_regression() {
      struct cleanup {
        std::array<slot, slot_limit> saved_slots{slots};
        std::array<evaluation_namespace, provider_count> saved_evaluations{evaluations};
        std::array<status, provider_count> saved_attempts{attempt_status[0].load(), attempt_status[1].load()};
        bool saved_requested{requested.load()};
        status saved_reason{reason.load()};
        ~cleanup() {
          slots = saved_slots; evaluations = saved_evaluations;
          for (unsigned i = 0; i != provider_count; ++i) attempt_status[i] = saved_attempts[i];
          requested = saved_requested; reason = saved_reason;
        }
      } restore;
      slots = {}; evaluations = {}; requested = true;
      constexpr std::uint64_t producer = 401, consumer = 402, device = 403, source_native = 404;
      auto source = std::make_shared<source_reference>();
      source->cookie = 405; source->device_identity = device;
      source->desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
      source->desc.Width = 2228; source->desc.Height = 1256; source->desc.Format = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
      source->desc.MipLevels = source->desc.DepthOrArraySize = source->desc.SampleDesc.Count = 1;
      input metadata;
      metadata.source = source; metadata.provider = provider_kind::ngx;
      metadata.epoch = 701; metadata.sequence = 1; metadata.source_id = 702; metadata.tick = GetTickCount64();
      metadata.resource.native = source_native; metadata.resource.area = {1, 2, 2227, 1253};
      // State/projection are deliberately unavailable: preservation owns the
      // GPU copy; nomination only describes a valid same-device source.
      metadata.proof = sunshine_scene_depth::state_proof::unavailable;
      auto recording = std::make_shared<sunshine_native_command::recording_lifetime>();
      recording->cookie = 501;
      begin_evaluation(metadata.epoch, metadata.sequence, 0, metadata.provider, metadata.source_id);
      const auto ticket = record_nomination(metadata, source_native, 501, recording);
      const auto nominated = std::find_if(slots.begin(), slots.end(), [&](const auto &value) { return value.id == ticket; });
      if (!ticket || nominated == slots.end()) return false;
      auto &nomination = *nominated;
      if (nomination.texture || !nomination.shared_preservation || nomination.finished) return false;
      queue_state owner; owner.device_identity = device;
      const auto select = [&] { return select_provider(owner, consumer, 601, GetTickCount64()); };
      if (select().value) return false;
      finish(ticket, true);
      if (select().value) return false;
      std::array<bool, slot_limit> producers{}, consumers{};
      producers[static_cast<std::size_t>(nominated - slots.begin())] = true;
      if (needs_submission_fence(producers, consumers)) return false;
      producer_submitted(nomination, producer, 0, true);
      if (select().value != &nomination || nomination.producer_fence || producer_recording_retired(nomination)) return false;
      if (submission_status(nomination, consumer, device + 1, {}) != status::unsupported_queue) return false;
      packet described;
      describe_packet(nomination, consumer, described);
      if (!described.shared_preservation || described.texture || described.shader_resource || described.ownership ||
          described.resource_id != source->cookie || described.metadata.source != source || described.producer_queue != producer ||
          described.queue != consumer || described.width != 2228 || described.height != 1256 ||
          described.area.left != 1 || described.area.top != 2 || described.area.width != 2227 || described.area.height != 1253 ||
          described.srv_format != DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS) return false;
      if (reclaimable(nomination)) return false; // Keep latest successful nomination.
      retire_recording(nomination, 501);
      if (select().value != &nomination || reclaimable(nomination)) return false;
      ++metadata.sequence; metadata.tick = GetTickCount64();
      begin_evaluation(metadata.epoch, metadata.sequence, 0, metadata.provider, metadata.source_id);
      if (select().value || !reclaimable(nomination)) return false; // Never conceal the new attempt with old metadata.
      const auto next = record_nomination(metadata, source_native, 502, recording);
      if (!next || next == ticket || nomination.id != next) return false;
      finish(ticket, true); // A late result cannot complete a reused slot.
      if (nomination.finished) return false;
      producer_submitted(nomination, producer, 0, true);
      finish(next, false);
      if (select().value || !reclaimable(nomination)) return false;
      // Reset before actual submission invalidates the nomination, without
      // holding any GPU texture or fence forever.
      const auto discarded = record_nomination(metadata, source_native, 503, recording);
      retire_recording(nomination, 503); finish(discarded, true);
      if (!nomination.invalid || !reclaimable(nomination) || select().value) return false;
      const auto replayed = record_nomination(metadata, source_native, 504, recording);
      finish(replayed, true); producer_submitted(nomination, producer, 0, true);
      native_observer::command_identity commands[]{{999, 504}};
      producers = {}; consumers = {};
      if (!identify_submissions(slots, 1, commands, producers, consumers) || !nomination.invalid ||
          nomination.failure != capture_failure::replay || !reclaimable(nomination) || needs_submission_fence(producers, consumers)) return false;
      slots[1].id = 1; producers[1] = true;
      if (!needs_submission_fence(producers, consumers)) return false; // Mixed native copies retain their fence.
      producers = {}; consumers[1] = true;
      return needs_submission_fence(producers, consumers);
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
      value.shared_preservation = false;
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
      if (out.pixel_ready || out.texture || out.ownership || out.shared_preservation || out.resource_id != 31 ||
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
        if (result_first) finish(value.id, false, capture_failure::evaluation_observation_changed);
        producer_submitted(value, native, fence, true);
        if (!result_first) finish(value.id, false, capture_failure::evaluation_observation_changed);
        if (!value.invalid || submitted_success(value, native) || value.failure != capture_failure::evaluation_observation_changed) return false;
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
        shared_preservation_regression() && preservation_owner_regression() && source_authority_regression();
    }
  }
#endif

}
