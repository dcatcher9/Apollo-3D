// SPDX-License-Identifier: GPL-3.0-only
// Part of exporter.cpp, the same translation unit (tools/reshade/README.md, Source map): the
// export ring's types and wire helpers, included inside exporter.cpp's anonymous namespace where
// they stood, so they keep internal linkage and their order. export_ring_class.inl holds the ring
// itself (export_ring); docs/reshade-sbs.md, GPU handoff contract, owns the protocol.

  bool exchange_state(wire::slot_t &slot, std::uint64_t generation, wire::slot_state desired, wire::slot_state expected) {
    const auto prior = wire::slot_control(generation, expected);
    return static_cast<std::uint64_t>(InterlockedCompareExchange64(reinterpret_cast<volatile LONG64 *>(&slot.control), static_cast<LONG64>(wire::slot_control(generation, desired)), static_cast<LONG64>(prior))) == prior;
  }

  void store_state(wire::slot_t &slot, std::uint64_t generation, wire::slot_state state) {
    InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&slot.control), static_cast<LONG64>(wire::slot_control(generation, state)));
  }

  std::uint64_t read_nonce(wire::shared_state_t &shared) {
    return static_cast<std::uint64_t>(InterlockedCompareExchange64(
      reinterpret_cast<volatile LONG64 *>(&shared.consumer_nonce),
      0,
      0
    ));
  }

  // The capabilities of the consumer answering nonce, zero for a replaced one
  // (wire::answered_capabilities).
  std::uint32_t read_capabilities(wire::shared_state_t &shared, std::uint64_t nonce) {
    const auto answered = static_cast<std::uint64_t>(InterlockedCompareExchange64(
      reinterpret_cast<volatile LONG64 *>(&shared.capability_nonce), 0, 0));
    const auto bits = static_cast<std::uint32_t>(InterlockedCompareExchange(
      reinterpret_cast<volatile LONG *>(&shared.consumer_capabilities), 0, 0));
    return wire::answered_capabilities(nonce, answered, bits);
  }

  // The protocol the consumer answering nonce declared, zero for a replaced
  // one or one that declared none (wire::answered_protocol).
  std::uint32_t read_consumer_protocol(wire::shared_state_t &shared, std::uint64_t nonce) {
    const auto answered = static_cast<std::uint64_t>(InterlockedCompareExchange64(
      reinterpret_cast<volatile LONG64 *>(&shared.capability_nonce), 0, 0));
    const auto protocol = static_cast<std::uint32_t>(InterlockedCompareExchange(
      reinterpret_cast<volatile LONG *>(&shared.consumer_protocol), 0, 0));
    return wire::answered_protocol(nonce, answered, protocol);
  }

  // The export transfer of an HDR swapchain for the answered consumer's
  // capabilities (docs/reshade-sbs.md, PQ wire transfer): 10-bit PQ for an
  // HDR10 source when the consumer accepts PQ, and for any HDR source, native
  // scRGB included, when it encodes HDR10 PQ (consumer_stream_pq, which
  // implies acceptance). Every other consumer gets FP16 scRGB.
  bool exports_pq(api::color_space source, std::uint32_t capabilities) {
    if (source == api::color_space::hdr10_pq)
      return (capabilities & (wire::consumer_accepts_pq | wire::consumer_stream_pq)) != 0;
    return source == api::color_space::scrgb && (capabilities & wire::consumer_stream_pq) != 0;
  }

  // C3: whether this Present's render may be exported to the consumer read at
  // export (consumer_pq: exports_pq for it). A native render chose its
  // transfer from the consumer it read before rendering (pq_requested), so
  // it is exported only while the consumer answers the same, in either
  // direction; otherwise the next Present renders for the new consumer and
  // no ring of the other transfer is created. A reference export cannot
  // render again: only its PQ output needs a consumer that takes PQ.
  bool transfer_current(bool native_render, bool pq_requested, wire::transfer output, bool consumer_pq) {
    return native_render ? consumer_pq == pq_requested : output != wire::transfer::pq || consumer_pq;
  }

  // The export pauses whose origin is the host, the game or the player rather
  // than a fault (tools/reshade/README.md, the readiness report): a host that
  // is absent, attached, replaced or without PQ, a game out of focus, reset,
  // presenting without a render or changing its colour space, and a
  // swapchain or frame not yet prepared. Each pause is logged as
  // "export inactive (reason, routine|fault)"; every other reason, a new one
  // included, is a fault. game3d_log_report.py reads the class and mirrors
  // this list for logs without it; its test reads the list from here.
  constexpr std::array<std::string_view, 11> routine_export_pauses{
    "no_consumer", "ring_replaced", "consumer_transfer_changed", "consumer_without_pq", "not_foreground",
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
  api::color_space export_space(wire::transfer transfer) {
    return transfer == wire::transfer::srgb ? api::color_space::srgb :
      transfer == wire::transfer::pq ? api::color_space::hdr10_pq : api::color_space::scrgb;
  }

  DXGI_FORMAT typed_format(DXGI_FORMAT format) {
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
  constexpr auto idle_ring_reuse = std::chrono::milliseconds(500);
  // Everything a generation's ring textures depend on, besides the consumer.
  bool same_ring(const source_t &a, const source_t &b) {
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
