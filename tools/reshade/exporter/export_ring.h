// SPDX-License-Identifier: GPL-3.0-only
// The export ring (docs/reshade-sbs.md, GPU handoff contract, owns the protocol):
// its wire helpers, the transfer and pause-class rules, the export colour and
// source descriptions, one ring's slots and GPU resources (generation_t), and
// export_ring, the slot owner and wire producer that publisher_t holds.
#pragma once
#include "../exporter_base.h"

namespace sunshine_exporter {
  inline bool exchange_state(wire::slot_t &slot, std::uint64_t generation, wire::slot_state desired, wire::slot_state expected) {
    const auto prior = wire::slot_control(generation, expected);
    return static_cast<std::uint64_t>(InterlockedCompareExchange64(reinterpret_cast<volatile LONG64 *>(&slot.control), static_cast<LONG64>(wire::slot_control(generation, desired)), static_cast<LONG64>(prior))) == prior;
  }

  inline void store_state(wire::slot_t &slot, std::uint64_t generation, wire::slot_state state) {
    InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&slot.control), static_cast<LONG64>(wire::slot_control(generation, state)));
  }

  inline std::uint64_t read_nonce(wire::shared_state_t &shared) {
    return static_cast<std::uint64_t>(InterlockedCompareExchange64(
      reinterpret_cast<volatile LONG64 *>(&shared.consumer_nonce),
      0,
      0
    ));
  }

  // The capabilities of the consumer answering nonce, zero for a replaced one
  // (wire::answered_capabilities).
  inline std::uint32_t read_capabilities(wire::shared_state_t &shared, std::uint64_t nonce) {
    const auto answered = static_cast<std::uint64_t>(InterlockedCompareExchange64(
      reinterpret_cast<volatile LONG64 *>(&shared.capability_nonce), 0, 0));
    const auto bits = static_cast<std::uint32_t>(InterlockedCompareExchange(
      reinterpret_cast<volatile LONG *>(&shared.consumer_capabilities), 0, 0));
    return wire::answered_capabilities(nonce, answered, bits);
  }

  // The protocol the consumer answering nonce declared, zero for a replaced
  // one or one that declared none (wire::answered_protocol).
  inline std::uint32_t read_consumer_protocol(wire::shared_state_t &shared, std::uint64_t nonce) {
    const auto answered = static_cast<std::uint64_t>(InterlockedCompareExchange64(
      reinterpret_cast<volatile LONG64 *>(&shared.capability_nonce), 0, 0));
    const auto protocol = static_cast<std::uint32_t>(InterlockedCompareExchange(
      reinterpret_cast<volatile LONG *>(&shared.consumer_protocol), 0, 0));
    return wire::answered_protocol(nonce, answered, protocol);
  }

  // The export transfer of an HDR swapchain for the answered consumer's
  // capabilities (docs/reshade-sbs.md, PQ wire transfer): 10-bit PQ for an
  // HDR10 source, whose every protocol 4 consumer accepts PQ, and for a native
  // scRGB source when the consumer encodes HDR10 PQ (consumer_stream_pq).
  // A native scRGB source stays FP16 scRGB for any other consumer.
  inline bool exports_pq(api::color_space source, std::uint32_t capabilities) {
    return source == api::color_space::hdr10_pq ||
      (source == api::color_space::scrgb && (capabilities & wire::consumer_stream_pq) != 0);
  }

  // C3: whether this Present's render may be exported to the consumer read at
  // export (consumer_pq: exports_pq for it). A native render chose its
  // transfer from the consumer it read before rendering (pq_requested), so
  // it is exported only while the consumer answers the same, in either
  // direction; otherwise the next Present renders for the new consumer and
  // no ring of the other transfer is created.
  inline bool transfer_current(bool pq_requested, bool consumer_pq) {
    return consumer_pq == pq_requested;
  }

  // The export pauses whose origin is the host, the game or the player rather
  // than a fault (tools/reshade/README.md, the readiness report): a host that
  // is absent, attached or replaced, a game out of focus, reset,
  // presenting without a render or changing its colour space, and a
  // swapchain or frame not yet prepared. Each pause is logged as
  // "export inactive (reason, routine|fault)"; every other reason, a new one
  // included, is a fault. game3d_log_report.py reads the class and mirrors
  // this list for logs without it; its test reads the list from here.
  inline constexpr std::array<std::string_view, 10> routine_export_pauses{
    "no_consumer", "ring_replaced", "consumer_transfer_changed", "not_foreground",
    "runtime_reset", "runtime_gone", "present_without_render", "color_space_changed", "swapchain_unknown",
    "frame_not_prepared"};
  constexpr const char *export_pause_class(std::string_view reason) {
    for (const auto routine : routine_export_pauses)
      if (routine == reason) return "routine";
    return "fault";
  }
  static_assert(std::string_view(export_pause_class("ring_replaced")) == "routine" &&
    std::string_view(export_pause_class("device_removed")) == "fault" && std::string_view(export_pause_class("")) == "fault");

  // The colour space a transfer's export is encoded in (the overlay's target).
  inline api::color_space export_space(wire::transfer transfer) {
    return transfer == wire::transfer::srgb ? api::color_space::srgb :
      transfer == wire::transfer::pq ? api::color_space::hdr10_pq : api::color_space::scrgb;
  }

  inline DXGI_FORMAT typed_format(DXGI_FORMAT format) {
    switch (format) {
      case DXGI_FORMAT_R8G8B8A8_TYPELESS:
      case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        return DXGI_FORMAT_R8G8B8A8_UNORM;
      case DXGI_FORMAT_B8G8R8A8_TYPELESS:
      case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        return DXGI_FORMAT_B8G8R8A8_UNORM;
      case DXGI_FORMAT_R10G10B10A2_TYPELESS:
        return DXGI_FORMAT_R10G10B10A2_UNORM;
      case DXGI_FORMAT_R16G16B16A16_TYPELESS:
        return DXGI_FORMAT_R16G16B16A16_FLOAT;
      default:
        return format;
    }
  }

  struct export_color_t {
    api::color_space input = api::color_space::unknown;
    wire::transfer output = wire::transfer::srgb;
  };



  struct source_t {
    api::resource resource {};
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    export_color_t color;
    std::uint64_t adapter = 0;
    HWND window = nullptr;
  };
  // A ring idle this long is reused for the next consumer (see generation_t::renew).
  inline constexpr auto idle_ring_reuse = std::chrono::milliseconds(500);
  // Everything a generation's ring textures depend on, besides the consumer.
  inline bool same_ring(const source_t &a, const source_t &b) {
    return a.width == b.width && a.height == b.height && a.format == b.format && a.adapter == b.adapter &&
      a.color.input == b.color.input && a.color.output == b.color.output && a.window == b.window;
  }

  // Why a presentation carries no depth, for the SBS output counters. unavailable: the
  // provider had none this Present (source switch, FG toggle, expiry). The reuse reasons
  // are a generated/pending frame the provider did give depth, refused here because the
  // previous presentation had none or resolved its scene for another source.
  enum class depth_gap : unsigned char { none, unavailable, reuse_after_gap, reuse_other_source };
  // One Game 3D publication's contribution to the SBS output counters.
  struct output_sample {
    bool fg = false;
    // With depth: rendered as the colour frame (scene not placed or strength
    // blend zero), or still fading in.
    bool scene_flat = false, scene_fading = false;
    bool depth_ready = false, reused_depth = false;
    depth_gap gap = depth_gap::none;
  };
  // Own native objects rather than ReShade wrappers, which disappear during runtime teardown.
  struct generation_t {
    api::device_api backend = api::device_api::d3d11;
    api::effect_runtime *runtime = nullptr;
    api::effect_runtime *owner_runtime = nullptr;
    bool owner_destroyed = false;
    std::uint64_t nonce = 0;
    std::uint64_t id = 0;
    std::uint64_t last_submitted = 0;
    std::uint64_t native_swapchain = 0;
    std::uint64_t pending_qpc = 0;
    float pending_ui_parallax_uv = 0.f;
    std::uint32_t pending_slot = wire::slot_count;
    bool signal_failed = false;
    bool pending_overlay = false;
    bool rendered_directly = false; // Logged once per generation.
    // When this ring stopped exporting (focus loss, consumer detach). An idle
    // ring can serve the next consumer under a new generation.
    std::chrono::steady_clock::time_point inactive_since{};
    // Diagnostic disposition travels with this copy until its slot is actually
    // published. A later effects callback must not relabel an earlier copy.
    std::optional<output_sample> pending_output;
    std::array<std::unique_ptr<sunshine::overlay::compositor_t>, wire::slot_count> overlays;
    // A D3D12 command list does not retain the resource referenced by a recorded copy.
    // Hold the source across effect reload/destruction until submission is completed.
    std::array<com_ptr<IUnknown>, wire::slot_count> submitted_sources;
    source_t source;
    std::array<handle_t, wire::slot_count> texture_handles;
    handle_t fence_handle;
    std::array<com_ptr<ID3D11Texture2D>, wire::slot_count> textures11;
    com_ptr<ID3D11Device5> device11;
    com_ptr<ID3D11DeviceContext4> context11;
    com_ptr<ID3D11Fence> fence11;
    std::array<com_ptr<ID3D12Resource>, wire::slot_count> textures12;
    com_ptr<ID3D12Device> device12;
    com_ptr<ID3D12CommandQueue> queue12;
    com_ptr<ID3D12Fence> fence12;

#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
    // SunshineGame3DTestTeardownOrder: named by the publisher member that
    // held it last (export_ring::note_rings).
    ~generation_t() {
      sunshine_game3d::test::teardown::generation_destroyed(this);
    }

#endif
    std::uint64_t completed() const {
      return fence11 ? fence11->GetCompletedValue() : fence12 ? fence12->GetCompletedValue() :
                                                                0;
    }

    bool finished() const {
      const auto value = completed();
      return value == UINT64_MAX || (pending_slot == wire::slot_count && !signal_failed && value >= last_submitted);
    }

    api::resource texture(std::uint32_t index) const {
      return {reinterpret_cast<std::uint64_t>(backend == api::device_api::d3d11 ? static_cast<void *>(textures11[index].get()) : static_cast<void *>(textures12[index].get()))};
    }

    // Unloading an add-on cannot run a background reaper after its DLL is gone. Preserve only
    // outstanding native allocations until process exit rather than wait on the game's GPU.
    void preserve_inflight_on_unload() {
      if (finished()) {
        return;
      }
      for (auto &texture : textures11) {
        texture.release();
      }
      for (auto &texture : textures12) {
        texture.release();
      }
      fence11.release();
      fence12.release();
      for (auto &source : submitted_sources) source.release();
      for (auto &overlay : overlays) overlay.release();
      log(reshade::log::level::warning, "Sunshine SBS: preserving one in-flight generation until process exit during add-on unload");
    }

    // A finished ring kept across a runtime reset (a resize) holds no back
    // buffer, which would fail ResizeBuffers: neither a submitted source nor
    // an overlay compositor (its target is the back buffer; one is created
    // again when the overlay next composes).
    void release_sources() {
      if (!finished()) return;
      for (auto &source : submitted_sources) source.reset();
      for (auto &overlay : overlays) overlay.reset();
    }

    // A finished ring idle for idle_ring_reuse serves a new consumer under a new
    // generation: slots are reset on publication and the consumer reopens the
    // same shared handles, instead of allocating another 2W x H ring.
    void renew(api::effect_runtime *owner, std::uint64_t consumer, std::uint64_t generation_id) {
      runtime = owner;
      nonce = consumer;
      id = generation_id;
      rendered_directly = false;
    }

    bool create(api::effect_runtime *owner, const source_t &input, std::uint64_t consumer, std::uint64_t generation_id) {
      runtime = owner;
      owner_runtime = owner;
      source = input;
      nonce = consumer;
      id = generation_id;
      native_swapchain = owner->get_native();
      return create_native(owner->get_device()->get_api(), owner->get_device()->get_native(), owner->get_command_queue()->get_native());
    }

    bool create_native(api::device_api renderer, std::uint64_t native_device_handle, std::uint64_t native_queue_handle) {
      backend = renderer;
      HRESULT hr = E_FAIL;
      if (backend == api::device_api::d3d11) {
        auto *native_device = reinterpret_cast<ID3D11Device *>(native_device_handle);
        auto *native_context = reinterpret_cast<ID3D11DeviceContext *>(native_queue_handle);
        if (FAILED(hr = native_device->QueryInterface(IID_PPV_ARGS(device11.put()))) || FAILED(hr = native_context->QueryInterface(IID_PPV_ARGS(context11.put())))) {
          log_hr("D3D11 shared fence interfaces", hr);
          return false;
        }
        if (FAILED(hr = device11->CreateFence(0, D3D11_FENCE_FLAG_SHARED, IID_PPV_ARGS(fence11.put()))) || FAILED(hr = fence11->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, fence_handle.put()))) {
          log_hr("D3D11 shared fence", hr);
          return false;
        }
        D3D11_TEXTURE2D_DESC desc {};
        desc.Width = source.width * 2;
        desc.Height = source.height;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = source.format;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        // Modern D3D11 NT sharing with a shared fence. No keyed-mutex waits are involved.
        desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
        for (std::uint32_t index = 0; index < wire::slot_count; ++index) {
          com_ptr<IDXGIResource1> shared;
          if (FAILED(hr = device11->CreateTexture2D(&desc, nullptr, textures11[index].put())) || FAILED(hr = textures11[index]->QueryInterface(IID_PPV_ARGS(shared.put()))) || FAILED(hr = shared->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, nullptr, texture_handles[index].put()))) {
            log_hr("D3D11 shared texture", hr);
            return false;
          }
        }
      } else if (backend == api::device_api::d3d12) {
        auto *native_device = reinterpret_cast<ID3D12Device *>(native_device_handle);
        auto *native_queue = reinterpret_cast<ID3D12CommandQueue *>(native_queue_handle);
        if (FAILED(hr = native_device->QueryInterface(IID_PPV_ARGS(device12.put()))) || FAILED(hr = native_queue->QueryInterface(IID_PPV_ARGS(queue12.put()))) || FAILED(hr = device12->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(fence12.put()))) || FAILED(hr = device12->CreateSharedHandle(fence12.get(), nullptr, GENERIC_ALL, nullptr, fence_handle.put()))) {
          log_hr("D3D12 shared fence", hr);
          return false;
        }
        D3D12_HEAP_PROPERTIES heap {};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC desc {};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = source.width * 2;
        desc.Height = source.height;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = source.format;
        desc.SampleDesc.Count = 1;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
        for (std::uint32_t index = 0; index < wire::slot_count; ++index) {
          if (FAILED(hr = device12->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(textures12[index].put()))) || FAILED(hr = device12->CreateSharedHandle(textures12[index].get(), nullptr, GENERIC_ALL, nullptr, texture_handles[index].put()))) {
            log_hr("D3D12 shared texture", hr);
            return false;
          }
        }
      } else {
        return false;
      }
      return true;
    }

    // The native renderer wrote this slot directly on the same command list;
    // its inputs are the renderer's own resources, retained by its fence.
    void adopt_rendered(std::uint32_t index, std::uint64_t sequence) {
      submitted_sources[index].reset();
      last_submitted = sequence;
    }

    // copy false: the overlay's composite pass, recorded in reshade_present,
    // writes every pixel of the slot from this same source (composite_ps
    // returns the source wherever the controls are transparent), so a copy
    // first would only be overwritten. The source is retained either way.
    void submit(api::command_list *commands, api::resource input, std::uint32_t index, std::uint64_t sequence,
        bool copy = true) {
      auto &submitted_source = submitted_sources[index];
      submitted_source.reset();
      auto *source_object = reinterpret_cast<IUnknown *>(input.handle);
      source_object->AddRef();
      *submitted_source.put() = source_object;
      const auto destination = texture(index);
      // Set this before appending commands: an exception cannot make recorded work appear idle.
      last_submitted = sequence;
      if (!copy) return;
      commands->barrier(input, api::resource_usage::shader_resource, api::resource_usage::copy_source);
      if (backend == api::device_api::d3d12) {
        commands->barrier(destination, api::resource_usage::general, api::resource_usage::copy_dest);
      }
      commands->copy_resource(input, destination);
      if (backend == api::device_api::d3d12) {
        commands->barrier(destination, api::resource_usage::copy_dest, api::resource_usage::general);
      }
      commands->barrier(input, api::resource_usage::copy_source, api::resource_usage::shader_resource);
      // The native overlay is drawn after techniques. Its composition must be included in
      // this same bounded submission before signalling or exposing the slot to Sunshine.
    }
  };

  // The export ring, the producer side of the wire: it owns the shared state's view, the
  // published metadata, the active, retired and cached generations, their slots and the ring
  // counters.
  //
  // publisher_t holds it as ring_, at the former position of the active generation, and calls
  // it only under publisher_t's mutex_. It has no user-declared destructor: the unload steps
  // run from publisher_t's destructor (unload, unmap), and its generations are destroyed with
  // ring_ (cached, retired, then active), after parked_ and before overlays_, as before.
  // What it needs of the publisher comes as callables: release_views(texture) drops the views
  // renderers cached of a released ring's slot, and count_output(runtime, sample, now) adds a
  // published slot's sample to its runtime's SBS output counters.
  class export_ring {
    friend class publisher_t;
#ifdef SUNSHINE_SBS_TEST
    friend struct publisher_tests;
#endif

  public:
    // Add-on unload (publisher_t's destructor, unless the process is
    // stopping): hosts see the identity alone, and each generation with
    // unfinished GPU work keeps its native allocations until process exit.
    void unload() {
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
      namespace teardown = sunshine_game3d::test::teardown;
#endif
      if (shared_) {
        publish(identity_);
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
        teardown::note(teardown::step::identity_published);
#endif
      }
      if (generation_) {
        generation_->preserve_inflight_on_unload();
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
        teardown::note(teardown::step::preserved, teardown::ring_owner::active);
#endif
      }
      if (retired_) {
        retired_->preserve_inflight_on_unload();
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
        teardown::note(teardown::step::preserved, teardown::ring_owner::retired);
#endif
      }
      if (cached_ring_) {
        cached_ring_->preserve_inflight_on_unload();
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
        teardown::note(teardown::step::preserved, teardown::ring_owner::cached);
#endif
      }
    }

    // The shared state's view ends with the publisher (its destructor); the
    // mapping is the publisher's.
    void unmap() {
      if (shared_) {
        UnmapViewOfFile(shared_);
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
        namespace teardown = sunshine_game3d::test::teardown;
        teardown::note(teardown::step::unmapped);
#endif
      }
    }

    // The ring part of frame(): gives the consumer `nonce` this runtime's ring
    // for `source`. The current generation serves while it matches; otherwise
    // (ring_replaced) the idle ring is renewed, the cached ring of this
    // transfer reused or a new ring created, and its metadata and handles
    // published. False: this Present exports nothing.
    template<class Views>
    bool serve(api::effect_runtime *runtime, const source_t &source, std::uint64_t nonce, api::device_api backend,
        const Views &release_views) {
      const bool replace = !generation_ || generation_->runtime != runtime || generation_->nonce != nonce ||
                           !same_ring(generation_->source, source);
      if (replace) {
        if (generation_) {
          deactivate(generation_->runtime, "ring_replaced");
        }
        retire_destroyed_generation(release_views);
        if (generation_ && !generation_->finished()) {
          return false;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now < retry_at_) {
          return false;
        }
        if (next_generation_ >= wire::max_generation) {
          deactivate(runtime, "generation_limit");
          return false;
        }
        // An abandoned consumer read of a reused slot finishes within frames;
        // a ring idle this long only ever served consumers that are gone. A
        // consumer that restarts while exporting still gets fresh resources.
        // finished() is also true for a removed device; such a ring is never reused.
        const bool reuse = generation_ && !generation_->owner_destroyed && generation_->owner_runtime == runtime &&
          generation_->completed() != UINT64_MAX &&
          generation_->native_swapchain == runtime->get_native() && same_ring(generation_->source, source) &&
          generation_->inactive_since != std::chrono::steady_clock::time_point{} && now - generation_->inactive_since >= idle_ring_reuse;
        // The cached ring of the other colour transfer (item 6: one entry,
        // released after cached_ring_idle) serves a toggle back the same way.
        const bool cached = !reuse && cached_ring_ && !cached_ring_->owner_destroyed && cached_ring_->owner_runtime == runtime &&
          cached_ring_->completed() != UINT64_MAX && cached_ring_->finished() &&
          cached_ring_->native_swapchain == runtime->get_native() && same_ring(cached_ring_->source, source) &&
          now - cached_ring_->inactive_since >= idle_ring_reuse;
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
        note_rings();  // Names a ring the assignments below destroy.
#endif
        // A finished ring that is not reused now stays cached for a toggle back.
        const auto keep_outgoing = [&] {
          if (generation_ && generation_->finished() && !generation_->owner_destroyed && generation_->completed() != UINT64_MAX) {
            if (generation_->inactive_since == std::chrono::steady_clock::time_point{}) generation_->inactive_since = now;
            cached_ring_ = std::move(generation_);
            cached_ring_since_ = now;
          }
        };
        if (reuse) {
          generation_->renew(runtime, nonce, ++next_generation_);
        } else if (cached) {
          auto previous = std::move(cached_ring_);
          keep_outgoing();
          generation_ = std::move(previous);
          generation_->renew(runtime, nonce, ++next_generation_);
        } else {
          auto next = std::make_unique<generation_t>();
          if (!next->create(runtime, source, nonce, ++next_generation_)) {
            deactivate(runtime, "ring_creation_failed");
            retry_at_ = now + std::chrono::seconds(2);
            return false;
          }
          keep_outgoing();
          generation_ = std::move(next);
        }
        wire::metadata_t metadata = identity_;
        set_source(metadata, source);
        metadata.generation = generation_->id;
        metadata.accepted_consumer_nonce = nonce;
        for (std::uint32_t index = 0; index < wire::slot_count; ++index) {
          metadata.texture_handles[index] = reinterpret_cast<std::uint64_t>(generation_->texture_handles[index].get());
        }
        metadata.ready_fence_handle = reinterpret_cast<std::uint64_t>(generation_->fence_handle.get());
        publish(metadata, true, runtime);
        char message[256];
        std::snprintf(message, sizeof(message), "Sunshine SBS: generation %llu, %ux%u full SBS, DXGI %u, D3D%u, %s, protocol %u (source color %u)%s", static_cast<unsigned long long>(metadata.generation), metadata.packed_width, metadata.packed_height, metadata.dxgi_format, backend == api::device_api::d3d11 ? 11u : 12u, metadata.color_transfer == wire::transfer::scrgb ? "scRGB" : metadata.color_transfer == wire::transfer::pq ? "PQ HDR10" : "sRGB", metadata.protocol_version, static_cast<unsigned>(source.color.input), reuse ? ", reusing the idle export ring" : cached ? ", reusing the cached export ring of this transfer" : "");
        log(reshade::log::level::info, message);
      }
      return true;
    }

    // Only a consumer that declared this export protocol is answered
    // (docs/reshade-sbs.md, GPU handoff contract): a host of another release
    // is refused by name, once per request, and gets no ring. Zero is a
    // request that declared nothing, or one being replaced (its successor's
    // declaration was read).
    bool consumer_speaks_protocol(api::effect_runtime *runtime, std::uint64_t nonce) {
      const auto protocol = read_consumer_protocol(*shared_, nonce);
      if (protocol == wire::version) return true;
      if (protocol && refused_nonce_ != nonce) {
        refused_nonce_ = nonce;
        char message[256];
        std::snprintf(message, sizeof(message), "Sunshine SBS: refusing a host that speaks export protocol %u; this add-on speaks protocol %u (%u export slots). Install the Sunshine 3D host and add-on of the same release; no stereo export to this host.", protocol, wire::version, wire::slot_count);
        log(reshade::log::level::warning, message);
      }
      deactivate(runtime, protocol ? "consumer_protocol" : "consumer_undeclared");
      return false;
    }

    // Claims a slot for this Present's export (docs/reshade-sbs.md, GPU handoff
    // contract): a free slot first, else the oldest ready slot that a newer
    // completed frame supersedes. The consumer claims the newest ready frame
    // whose fence passed, so the newest completed frame of the ready and
    // reading slots is the one it claims next (or holds) and is never written
    // over, even while a newer write is still queued: a consumer that comes
    // late still finds it, instead of nothing until the queued write lands. A
    // slot is reusable once its previous GPU write completed: a ready slot
    // whose write is still queued is never reused either. A consumer may
    // discard a frame without reading it, so a free slot also waits for its
    // write to complete (every free slot of the current generation has:
    // reset, or freed by a consumer after its fence passed).
    // overwrote: a ready slot was claimed.
    std::uint32_t acquire_slot(std::uint64_t completed, bool *overwrote = nullptr) {
      if (overwrote) *overwrote = false;
      // A recorded copy must be fenced before admitting the next. Submitted
      // work is bounded by the existing ring, not a global one-copy gate.
      if (!generation_ || generation_->signal_failed || completed == UINT64_MAX ||
          generation_->pending_slot != wire::slot_count) return wire::slot_count;
      const auto id = generation_->id;
      std::array<wire::slot_state, wire::slot_count> states{};
      std::array<bool, wire::slot_count> current{};
      std::uint64_t newest_completed = 0;
      for (std::uint32_t index = 0; index < wire::slot_count; ++index) {
        const auto control = static_cast<std::uint64_t>(InterlockedCompareExchange64(
          reinterpret_cast<volatile LONG64 *>(&shared_->slots[index].control), 0, 0));
        current[index] = wire::control_generation(control) == id;
        states[index] = wire::control_state(control);
        if (current[index] && (states[index] == wire::slot_state::ready || states[index] == wire::slot_state::reading) &&
            shared_->slots[index].sequence <= completed)
          newest_completed = std::max(newest_completed, shared_->slots[index].sequence);
      }
      const auto written = [&](std::uint32_t index) {
        return current[index] && completed >= shared_->slots[index].sequence;
      };
      for (std::uint32_t offset = 0; offset < wire::slot_count; ++offset) {
        const auto index = (next_slot_ + offset) % wire::slot_count;
        if (states[index] == wire::slot_state::free && written(index) &&
            exchange_state(shared_->slots[index], id, wire::slot_state::writing, wire::slot_state::free)) return index;
      }
      // Completed ready slots oldest first; a claim the consumer won meanwhile moves on.
      std::array<std::uint32_t, wire::slot_count> order{};
      for (std::uint32_t index = 0; index < wire::slot_count; ++index) order[index] = index;
      std::sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
        return shared_->slots[a].sequence < shared_->slots[b].sequence;
      });
      for (const auto index : order) {
        if (states[index] != wire::slot_state::ready || !written(index) || shared_->slots[index].sequence >= newest_completed) continue;
        if (exchange_state(shared_->slots[index], id, wire::slot_state::writing, wire::slot_state::ready)) {
          if (overwrote) *overwrote = true;
          return index;
        }
      }
      return wire::slot_count;
    }

    template<class Views>
    void retire_destroyed_generation(const Views &release_views) {
      if (retired_ && retired_->finished()) {
        release_ring(retired_, release_views);
      }
      if (!generation_ || !generation_->owner_destroyed) {
        return;
      }
      if (generation_->finished()) {
        release_ring(generation_, release_views);
        return;
      }
      if (!retired_) {
        retired_ = std::move(generation_);
        log(reshade::log::level::warning, "Sunshine SBS: retaining a destroyed runtime's pending generation; successor runtime may export");
      } else if (!retirement_limit_logged_) {
        retirement_limit_logged_ = true;
        log(reshade::log::level::error, "Sunshine SBS: two runtime generations have unconfirmed GPU work; restart the game to resume export");
      }
    }

    // One-shot invalidation also covers identity metadata published before a host connects.
    // The reason is logged once, when the export actually stops; only then
    // does this return true.
    bool deactivate(api::effect_runtime *runtime, const char *reason = "stopped") {
      if (!runtime) {
        return false;
      }
      const bool stopped = published_runtime_ == runtime;
      if (stopped) {
        publish(identity_);
        char message[160];
        std::snprintf(message, sizeof(message), "Sunshine SBS: export inactive (%s, %s); waiting for a valid focused technique", reason,
          export_pause_class(reason));
        log(reshade::log::level::info, message);
      }
      if (generation_ && generation_->runtime == runtime) {
        for (auto &overlay : generation_->overlays) if (overlay) overlay->cancel();
        generation_->runtime = nullptr;
        generation_->inactive_since = std::chrono::steady_clock::now();
      }
      return stopped;
    }

    // Publishes the pending export, then queues its fence signal behind the
    // pack and overlay already submitted on the generation's queue (D3D11
    // flushes after it). Ready first: the consumer claims a slot only once the
    // fence reached its sequence, so a ready slot is never read early and
    // every fence wake finds its slot ready. A recording retired by reload,
    // focus or nonce changes, or whose overlay composite was never recorded,
    // is still fenced but never published. A failed signal withdraws the slot
    // and deactivates the export (runtime: the reshade_present caller, null
    // from finish_present).
    template<class Counts>
    void signal_pending(api::effect_runtime *runtime, const Counts &count_output) {
      auto &g = *generation_;
      const auto index = std::exchange(g.pending_slot, wire::slot_count);
      const bool composed = !std::exchange(g.pending_overlay, false);
      if (!composed && g.overlays[index]) g.overlays[index]->cancel();
      const bool publish = composed && g.runtime && published_runtime_ == g.runtime && read_nonce(*shared_) == g.nonce;
      if (publish) complete_slot(index, g.pending_qpc, count_output);
      HRESULT hr = E_FAIL;
      if (g.backend == api::device_api::d3d11) {
        hr = g.context11->Signal(g.fence11.get(), g.last_submitted);
        g.context11->Flush();
      } else {
        // The add-on's own fence (signal_own): never a game Signal to the
        // order hooks.
        hr = sunshine_streamline::native_observer::signal_own(g.queue12.get(), g.fence12.get(), g.last_submitted);
      }
      if (SUCCEEDED(hr)) return;
      g.signal_failed = true;
      if (publish) exchange_state(shared_->slots[index], g.id, wire::slot_state::writing, wire::slot_state::ready);
      log_hr(g.backend == api::device_api::d3d11 ? "D3D11 publication fence signal" : "D3D12 publication fence signal", hr);
      deactivate(runtime ? runtime : g.runtime, "fence_signal_failed");
    }

    // A ring that stopped exporting (deactivated: focus loss, no consumer)
    // and has no unfinished work is released after export_ring_idle, so a
    // stereo-to-mono switch frees its four 2W x H slots (265-530 MB at 4K); a
    // later export allocates a new ring. The cached ring of the other colour
    // transfer keeps its own cached_ring_idle.
    template<class Views>
    void release_idle_rings(const Views &release_views) {
      if ((!generation_ || generation_->runtime) && !cached_ring_) return;
      const auto now = std::chrono::steady_clock::now();
      if (generation_ && !generation_->runtime &&
          generation_->inactive_since != std::chrono::steady_clock::time_point{} &&
          now - generation_->inactive_since >= export_ring_idle && generation_->finished()) {
        log(reshade::log::level::info, "Sunshine SBS: released the idle export ring");
        release_ring(generation_, release_views);
      }
      if (cached_ring_ && now - cached_ring_since_ >= cached_ring_idle && cached_ring_->finished())
        release_ring(cached_ring_, release_views);
    }

    // Releases a ring that no pack follows. Renderers cache render-target
    // views of the slots they pack into (renderer::pack) and drop them only
    // at their next pack; a D3D11 view holds its texture, so they are dropped
    // here first or the slots' VRAM would stay allocated.
    template<class Views>
    void release_ring(std::unique_ptr<generation_t> &ring, const Views &release_views) {
      if (!ring) return;
      for (std::uint32_t index = 0; index < wire::slot_count; ++index) {
        const auto texture = ring->texture(index);
        if (!texture.handle) continue;
        release_views(texture);
      }
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
      note_rings();
#endif
      ring.reset();
    }

#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
    // SunshineGame3DTestTeardownOrder: the member that holds each ring now,
    // which names the ring when it is destroyed.
    void note_rings() const {
      sunshine_game3d::test::teardown::note_rings(generation_.get(), retired_.get(), cached_ring_.get());
    }
#endif

    template<class Counts>
    void complete_slot(std::uint32_t index, std::uint64_t qpc, const Counts &count_output) {
      const auto now = sunshine_game3d::clock::game_now();
      auto &slot = shared_->slots[index];
      slot.sequence = generation_->last_submitted;
      slot.qpc = qpc;
      slot.cursor_plane_flags = wire::cursor_plane_present;
      slot.ui_parallax_uv = generation_->pending_ui_parallax_uv;
      store_state(slot, generation_->id, wire::slot_state::ready);
      if (const auto sample = std::exchange(generation_->pending_output, std::nullopt))
        count_output(generation_->runtime, *sample, now);
    }

    void publish(const wire::metadata_t &metadata, bool reset_slots = false, api::effect_runtime *owner = nullptr) {
      InterlockedIncrement(reinterpret_cast<volatile LONG *>(&shared_->metadata_sequence));
      published_metadata_ = metadata;
      published_runtime_ = owner;
      MemoryBarrier();
      shared_->metadata = metadata;
      if (reset_slots) {
        for (auto &slot : shared_->slots) {
          slot.sequence = 0;
          slot.qpc = 0;
          slot.cursor_plane_flags = 0;
          slot.ui_parallax_uv = 0.f;
          store_state(slot, metadata.generation, wire::slot_state::free);
        }
      }
      MemoryBarrier();
      InterlockedIncrement(reinterpret_cast<volatile LONG *>(&shared_->metadata_sequence));
    }

    static void set_source(wire::metadata_t &metadata, const source_t &source) {
      metadata.window = reinterpret_cast<std::uint64_t>(source.window);
      metadata.adapter_luid = source.adapter;
      metadata.source_width = source.width;
      metadata.source_height = source.height;
      metadata.packed_width = source.width * 2;
      metadata.packed_height = source.height;
      metadata.dxgi_format = static_cast<std::uint32_t>(source.format);
      metadata.color_transfer = source.color.output;
      metadata.protocol_version = wire::version;
    }

  private:
    wire::shared_state_t *shared_ = nullptr;
    wire::metadata_t identity_;
    wire::metadata_t published_metadata_;
    api::effect_runtime *published_runtime_ = nullptr;
    std::unique_ptr<generation_t> generation_;
    // A missing finish_present after runtime destruction must not wedge the next runtime.
    // Bound this fallback to one retired ring plus the active ring; never allocate indefinitely.
    std::unique_ptr<generation_t> retired_;
    // The finished ring of the previous colour transfer or extent, reused on a
    // toggle back (item 6) and released once cached this long.
    std::unique_ptr<generation_t> cached_ring_;
    std::chrono::steady_clock::time_point cached_ring_since_{};
    static constexpr auto cached_ring_idle = std::chrono::seconds(60);
    static constexpr auto export_ring_idle = std::chrono::seconds(10);
    bool retirement_limit_logged_ = false;
    std::uint64_t next_generation_ = 0;
    std::uint32_t next_slot_ = 0;
    // The last request refused for another export protocol, named once.
    std::uint64_t refused_nonce_ = 0;
    std::chrono::steady_clock::time_point retry_at_ {};
  };
}  // namespace sunshine_exporter
