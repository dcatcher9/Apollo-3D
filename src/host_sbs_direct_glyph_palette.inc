// Included inside host_sbs_direct_glyph_gpu::detail. One estimator owns one private
// TensorRT context, one observation, and one committed raw history generation.
class palette_transport_t {
  using buffer_ptr = ComPtr<ID3D11Buffer>;
  using srv_ptr = ComPtr<ID3D11ShaderResourceView>;
  using uav_ptr = ComPtr<ID3D11UnorderedAccessView>;
  using sources_t = std::array<ID3D11ShaderResourceView *, 16>;
  using targets_t = std::array<ID3D11UnorderedAccessView *, 5>;
  using binding_t = std::pair<unsigned, ID3D11ShaderResourceView *>;

  struct field_t {
    buffer_ptr buffer;
    srv_ptr srv;
    uav_ptr uav;
    UINT capacity = 0;
  };

  struct history_t {
    field_t lines, raw;
  };

  enum stage_e {
    pack_rgb,
    pack_tile,
    export_roles,
    coarse,
    make_hist,
    smooth_hist,
    choose_colors,
    make_moments,
    smooth_moments,
    prototypes,
    decode_palette,
    begin_history,
    confirm_history,
    finish_windows,
    confirm_mask_history,
    publish_line,
    finish_history,
    stage_count
  };
  enum class phase_e {
    idle,
    plan,
    scoring,
    finishing,
    leased
  };
  phase_e phase_ = phase_e::idle;
  ComPtr<ID3D11Device> device_;
  ComPtr<ID3D11DeviceContext> ctx_;
  std::array<ComPtr<ID3D11ComputeShader>, stage_count> shaders_;
  buffer_ptr constants_, staged_sizes_;
  ComPtr<ID3D11Query> event_;
  field_t work_, sizes_, input_, logits_, roles_, low_, fine_, peak_, mean_;
  field_t hist_, lut_, temp_, palette_, winner_, moment_, squared_, h1_, h2_, near_, far_, parts_, proto_, alpha_;
  std::array<history_t, 2> history_;
  unsigned previous_ = 0;
  bool history_valid_ = false;
  ComPtr<ID3D11Texture2D> old_rgb_, protection_;
  srv_ptr old_rgb_srv_, protection_srv_;
  uav_ptr protection_uav_;
  UINT width_ = 0, height_ = 0;
  request_t job_ {}, prior_job_ {};
  buffer_ptr submitted_work_;
  srv_ptr source_, caps_;
  std::array<std::array<UINT, 2>, 8> sizes_cpu_ {};
  UINT slot_ = 0, tile_start_ = 0, w28_ = 0, w84_ = 0, step_ = 1, hist_scale_ = 0;
  bool has_lines_ = false;
  cuda_driver_api &cuda_ = cuda_driver_api::get();
  CUcontext primary_ = nullptr;
  CUdevice adapter_ {};
  CUstream stream_ = nullptr;
  nvinfer1::IExecutionContext *execution_ = nullptr;
  std::array<CUgraphicsResource, 2> interop_ {};
  bool cuda_outstanding_ = false;

  static UINT div_up(UINT n, UINT d) {
    return (n + d - 1u) / d;
  }

  history_t &next() {
    return history_[1u - previous_];
  }

  static bool same_geometry(const request_t &a, const request_t &b) {
    return a.width == b.width && a.band_height == b.band_height && a.crop_top == b.crop_top &&
           a.origin_x == b.origin_x && a.origin_y == b.origin_y && a.generation == b.generation &&
           a.color_domain == b.color_domain && a.line_center_twice_source == b.line_center_twice_source &&
           a.frame > b.frame;
  }

  void ensure(field_t &f, UINT count, UINT stride) {
    if (f.capacity >= count) {
      return;
    }
    const std::uint64_t bytes = std::uint64_t(count) * stride;
    require(count && bytes <= UINT_MAX, "Native glyph scratch exceeds a D3D buffer");
    field_t replacement;
    D3D11_BUFFER_DESC desc {static_cast<UINT>(bytes), D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS, 0u, D3D11_RESOURCE_MISC_BUFFER_STRUCTURED, stride};
    require(SUCCEEDED(device_->CreateBuffer(&desc, nullptr, &replacement.buffer)) && SUCCEEDED(device_->CreateShaderResourceView(replacement.buffer.Get(), nullptr, &replacement.srv)) && SUCCEEDED(device_->CreateUnorderedAccessView(replacement.buffer.Get(), nullptr, &replacement.uav)), "Allocate native glyph scratch");
    replacement.capacity = count;
    f = std::move(replacement);
  }

  buffer_ptr make_buffer(D3D11_BUFFER_DESC desc) {
    buffer_ptr b;
    require(SUCCEEDED(device_->CreateBuffer(&desc, nullptr, &b)), "Allocate native glyph scheduling buffer");
    return b;
  }

  void band(UINT w, UINT h) {
    if (width_ == w && height_ == h) {
      return;
    }
    history_valid_ = false;
    D3D11_TEXTURE2D_DESC desc {w, h, 1, 1, DXGI_FORMAT_R16G16B16A16_FLOAT, {1, 0}, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0, 0};
    require(SUCCEEDED(device_->CreateTexture2D(&desc, nullptr, old_rgb_.ReleaseAndGetAddressOf())) && SUCCEEDED(device_->CreateShaderResourceView(old_rgb_.Get(), nullptr, old_rgb_srv_.ReleaseAndGetAddressOf())), "Create native glyph history texture");
    desc.Format = DXGI_FORMAT_R32_UINT;
    desc.BindFlags |= D3D11_BIND_UNORDERED_ACCESS;
    require(SUCCEEDED(device_->CreateTexture2D(&desc, nullptr, protection_.ReleaseAndGetAddressOf())) && SUCCEEDED(device_->CreateShaderResourceView(protection_.Get(), nullptr, protection_srv_.ReleaseAndGetAddressOf())) && SUCCEEDED(device_->CreateUnorderedAccessView(protection_.Get(), nullptr, protection_uav_.ReleaseAndGetAddressOf())), "Create native glyph publication texture");
    width_ = w;
    height_ = h;
  }

  void dispatch(stage_e stage, const std::array<UINT, 8> &cb, UINT gx, UINT gy, std::initializer_list<binding_t> inputs, targets_t outputs) {
    sources_t reads {};
    for (const auto &[index, srv] : inputs) {
      reads[index] = srv;
    }
    if (stage <= decode_palette) {
      reads[11] = next().lines.srv.Get();
    }
    ctx_->UpdateSubresource(constants_.Get(), 0, nullptr, cb.data(), 0, 0);
    ctx_->CSSetConstantBuffers(0, 1, constants_.GetAddressOf());
    ctx_->CSSetShaderResources(0, UINT(reads.size()), reads.data());
    ctx_->CSSetUnorderedAccessViews(0, UINT(outputs.size()), outputs.data(), nullptr);
    ctx_->CSSetShader(shaders_[stage].Get(), nullptr, 0);
    ctx_->Dispatch(gx, gy, 1);
    reads.fill(nullptr);
    outputs.fill(nullptr);
    ctx_->CSSetShaderResources(0, UINT(reads.size()), reads.data());
    ctx_->CSSetUnorderedAccessViews(0, UINT(outputs.size()), outputs.data(), nullptr);
  }

  std::array<UINT, 8> schedule() const {
    return {width_, height_, UINT(job_.token), UINT(job_.token >> 32u), UINT(source_ != nullptr), UINT(history_valid_), slot_, job_.line_center_twice_source};
  }

  void stage(stage_e entry, UINT arg, UINT gx, UINT gy, std::initializer_list<binding_t> inputs, field_t &destination, field_t *second = nullptr) {
    dispatch(entry, {slot_, w28_, w84_, arg, hist_scale_, 0, 0, 0}, gx, gy, inputs, {destination.uav.Get(), second ? second->uav.Get() : nullptr});
  }

  void fence() {
    ctx_->End(event_.Get());
    // The file worker may have no further D3D commands to submit until the scheduler wakes.
    ctx_->Flush();
  }

  void encode() {
    constexpr UINT count = 2;
    cuda_outstanding_ = true;
    check_cuda(cuda_.cuGraphicsMapResources(count, interop_.data(), stream_), "Map native glyph encoder bindings");
    std::array<CUdeviceptr, count> addr {};
    for (UINT i = 0; i < count; ++i) {
      std::size_t bytes = 0;
      check_cuda(cuda_.cuGraphicsResourceGetMappedPointer(&addr[i], &bytes, interop_[i]), "Map native glyph tensor");
      require(addr[i] && bytes == sizeof(float) * (i == 0 ? tile_input_words : tile_output_words), "Native glyph encoder tensor extent mismatch");
    }
    require(execution_->setTensorAddress("rgb", reinterpret_cast<void *>(addr[0])) && execution_->setTensorAddress("logits", reinterpret_cast<void *>(addr[1])) && execution_->enqueueV3(stream_), "Submit mixed native glyph encoder");
    check_cuda(cuda_.cuGraphicsUnmapResources(count, interop_.data(), stream_), "Return native glyph tensors to D3D");
  }

  void begin_line(UINT w, UINT h) {
    w28_ = div_up(w * 28u, h);
    w84_ = div_up(w * 84u, h);
    // Match the reference's ties-to-even integer sampling step.
    step_ = std::max(1u, UINT(std::nearbyint(double(h) / 96.0)));
    hist_scale_ = std::min(65536u, 0x7fffffffu / (div_up(w, step_) * div_up(h, step_)));
    const UINT n = w84_ * 84u;
    ensure(low_, w28_ * 28u, 4);
    ensure(roles_, w28_ * 28u, 16);
    ensure(peak_, w28_ * 28u, 16);
    ensure(mean_, w28_ * 28u, 16);
    ensure(fine_, n, 4);
    for (auto *field : {&moment_, &squared_, &h1_, &h2_, &near_, &far_}) {
      ensure(*field, n, 16);
    }
    ensure(parts_, 2u * div_up(n, 128), 16);
    ensure(alpha_, w * h, 4);
    const std::array<UINT, 4> zero {};
    ctx_->ClearUnorderedAccessViewUint(hist_.uav.Get(), zero.data());
    stage(pack_rgb, 28, div_up(w28_, 16), 4, {{0, source_.Get()}}, low_);
    stage(pack_rgb, 84, div_up(w84_, 16), 11, {{0, source_.Get()}}, fine_);
  }

  void finish_line(UINT w, UINT h) {
    // Use the same sampled pixel centers for the color counts and their areas.
    const UINT pad = std::max(1u, UINT(std::nearbyint(double(h) / 8.0)));
    const auto interior_extent = [&](UINT length) {
      return length > 2u * pad ? div_up(length - pad, step_) - div_up(pad, step_) : 0u;
    };
    const UINT inside = interior_extent(w) * interior_extent(h);
    const UINT total = div_up(w, step_) * div_up(h, step_);
    stage(coarse, 0, div_up(w28_, 16), 4, {{1, roles_.srv.Get()}}, peak_, &mean_);
    stage(make_hist, pad, div_up(div_up(w, step_), 16), div_up(div_up(h, step_), 8), {{0, source_.Get()}, {1, roles_.srv.Get()}}, hist_);
    stage(smooth_hist, 0, 39, 1, {{9, hist_.srv.Get()}}, lut_);
    stage(smooth_hist, 1, 39, 1, {{10, lut_.srv.Get()}}, temp_);
    stage(smooth_hist, 2, 39, 1, {{10, temp_.srv.Get()}}, lut_, &palette_);
    dispatch(choose_colors, {slot_, w28_, w84_, 0, hist_scale_, total - inside, inside, 0},
      1, 1, {{4, palette_.srv.Get()}, {10, lut_.srv.Get()}}, {winner_.uav.Get()});
    stage(make_moments, 0, div_up(w84_, 16), 11, {{1, roles_.srv.Get()}, {8, fine_.srv.Get()}}, moment_, &squared_);
    for (UINT radius : {6u, 17u}) {
      stage(smooth_moments, radius, div_up(w84_, 16), 11, {{12, moment_.srv.Get()}, {13, squared_.srv.Get()}}, h1_, &h2_);
      stage(smooth_moments, 64u + radius, div_up(w84_, 16), 11, {{12, h1_.srv.Get()}, {13, h2_.srv.Get()}}, radius == 6u ? near_ : far_);
    }
    const UINT groups = div_up(w84_ * 84u, 128);
    stage(prototypes, 0, std::min(groups, 65535u), div_up(groups, 65535u), {{1, roles_.srv.Get()}, {8, fine_.srv.Get()}}, parts_);
    stage(prototypes, 1, 1, 1, {{15, parts_.srv.Get()}}, proto_);
    stage(decode_palette, 0, div_up(w, 16), div_up(h, 8), {{0, source_.Get()}, {1, roles_.srv.Get()}, {2, peak_.srv.Get()}, {3, mean_.srv.Get()}, {4, palette_.srv.Get()}, {5, winner_.srv.Get()}, {6, near_.srv.Get()}, {8, fine_.srv.Get()}, {14, far_.srv.Get()}, {15, proto_.srv.Get()}}, alpha_);
    auto &old = history_[previous_];
    if (history_valid_) {
      dispatch(confirm_mask_history, schedule(), 1, 1, {{1, old.lines.srv.Get()}, {2, old_rgb_srv_.Get()}, {3, source_.Get()}, {4, old.raw.srv.Get()}, {5, alpha_.srv.Get()}}, {nullptr, next().lines.uav.Get()});
    }
    dispatch(publish_line, schedule(), div_up(w, 16), div_up(h, 8), {{1, old.lines.srv.Get()}, {2, old_rgb_srv_.Get()}, {3, source_.Get()}, {4, old.raw.srv.Get()}, {5, alpha_.srv.Get()}, {6, next().lines.srv.Get()}}, {work_.uav.Get(), nullptr, nullptr, next().raw.uav.Get(), protection_uav_.Get()});
  }

  void score_chunk() {
    const current_cuda_context_t guard(cuda_, primary_);
    const compute_state_t<16, 5> saved(ctx_.Get());
    UINT budget = tile_capacity;
    while (slot_ < sizes_cpu_.size() && budget) {
      const auto [w, h] = sizes_cpu_[slot_];
      if (!w) {
        ++slot_;
        continue;
      }
      if (tile_start_ == 0) {
        begin_line(w, h);
      }
      stage(pack_tile, tile_start_, 32, 18, {{7, low_.srv.Get()}}, input_);
      encode();
      stage(export_roles, tile_start_, div_up(std::min(440u, w28_ - tile_start_), 16), 4, {{12, logits_.srv.Get()}}, roles_);
      tile_start_ += 440u;
      --budget;
      if (tile_start_ >= w28_) {
        finish_line(w, h);
        ++slot_;
        tile_start_ = 0;
      }
    }
    while (slot_ < sizes_cpu_.size() && sizes_cpu_[slot_][0] == 0) {
      ++slot_;
    }
    if (slot_ == sizes_cpu_.size()) {
      dispatch(finish_history, schedule(), 1, 1, {}, {work_.uav.Get(), next().lines.uav.Get()});
      if (has_lines_) {
        ComPtr<ID3D11Resource> src;
        source_->GetResource(&src);
        ctx_->CopyResource(old_rgb_.Get(), src.Get());
      }
      phase_ = phase_e::finishing;
    } else {
      phase_ = phase_e::scoring;
    }
    fence();
  }

  void drop() {
    phase_ = phase_e::idle;
    submitted_work_.Reset();
    source_.Reset();
    caps_.Reset();
  }

  poll_status_e publish(result_t &result) {
    ctx_->CopyResource(submitted_work_.Get(), work_.buffer.Get());
    result.alpha8 = protection_srv_;
    phase_ = phase_e::leased;
    return poll_status_e::ready;
  }

public:
  void prepare(ID3D11Device *device, ID3D11DeviceContext *context, const std::filesystem::path &root, CUcontext primary, nvinfer1::IExecutionContext *execution) {
    require(device && context && primary && execution, "Native glyph transport requires an execution lease");
    device_ = device;
    ctx_ = context;
    ComPtr<ID3D11DeviceContext> immediate;
    device->GetImmediateContext(&immediate);
    require(context == immediate.Get(), "Native glyph transport requires its device's immediate context");
    const auto &specs = cache::direct_glyph_palette_specs;
    static_assert(specs.size() == stage_count);
    const auto sources = cache::snapshot_sources(root, specs);
    require(sources && cache::source_closure_sha256(sources) == cache::direct_glyph_palette_source_closure_sha256, "Native glyph shader closure mismatch");
    for (UINT i = 0; i < stage_count; ++i) {
      const auto code = cache::get(sources, specs[i]);
      require(code && SUCCEEDED(device->CreateComputeShader(code->data(), code->size(), nullptr, &shaders_[i])), "Prepare native glyph shader program");
    }
    execution_ = execution;
    require(execution_->getMaxOutputSize("logits") == sizeof(float) * tile_output_words, "Native glyph requires the mixed H144 W512 encoder");
    require(cuda_.is_valid() && cuda_.cuD3D11GetDevice && cuda_.cuDevicePrimaryCtxRetain && cuda_.cuDevicePrimaryCtxRelease, "Native glyph CUDA interop is unavailable");
    ComPtr<IDXGIDevice> dx;
    ComPtr<IDXGIAdapter> adapter;
    require(SUCCEEDED(device_.As(&dx)) && SUCCEEDED(dx->GetAdapter(&adapter)), "Read native glyph adapter");
    check_cuda(cuda_.cuD3D11GetDevice(&adapter_, adapter.Get()), "Select native glyph CUDA adapter");
    check_cuda(cuda_.cuDevicePrimaryCtxRetain(&primary_, adapter_), "Retain native glyph CUDA primary");
    require(primary_ == primary, "Native glyph context belongs to a different CUDA primary");
    const current_cuda_context_t guard(cuda_, primary_);
    check_cuda(cuda_.cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING), "Create bounded native glyph stream");
    ensure(input_, tile_input_words, 4);
    ensure(logits_, tile_output_words, 4);
    check_cuda(cuda_.cuGraphicsD3D11RegisterResource(&interop_[0], input_.buffer.Get(), 0), "Register native glyph RGB");
    check_cuda(cuda_.cuGraphicsD3D11RegisterResource(&interop_[1], logits_.buffer.Get(), 0), "Register native glyph logits");
    constants_ = make_buffer({32, D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0});
    staged_sizes_ = make_buffer({64, D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ, 0, 0});
    work_.buffer = make_buffer({work_bytes, D3D11_USAGE_DEFAULT, D3D11_BIND_UNORDERED_ACCESS, 0, D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS, 0});
    D3D11_UNORDERED_ACCESS_VIEW_DESC v {};
    v.Format = DXGI_FORMAT_R32_TYPELESS;
    v.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    v.Buffer.NumElements = work_bytes / 4u;
    v.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
    require(SUCCEEDED(device->CreateUnorderedAccessView(work_.buffer.Get(), &v, &work_.uav)), "Create native glyph Work view");
    D3D11_QUERY_DESC event {D3D11_QUERY_EVENT, 0};
    require(SUCCEEDED(device->CreateQuery(&event, &event_)), "Create native glyph nonblocking completion fence");
    ensure(sizes_, 8, 8);
    for (auto &generation : history_) {
      ensure(generation.lines, 8, 64);
      ensure(generation.raw, 1, 4);
    }
    // Four role-vote counts and two spatial counts per RGB bin.
    ensure(hist_, 6 * 4913, 4);
    ensure(lut_, 2 * 4913, 16);
    ensure(temp_, 2 * 4913, 16);
    ensure(palette_, 4913, 16);
    ensure(winner_, 2, 4);
    ensure(proto_, 2, 16);
  }

  bool busy() const noexcept {
    return phase_ != phase_e::idle;
  }

  ID3D11Buffer *committed_lines() const noexcept {
    return history_[previous_].lines.buffer.Get();
  }

  void submit(const request_t &request) {
    require(!busy() && request.work && request.detector_caps && request.token && request.width && request.width <= 5120u && request.band_height && request.band_height <= 5120u, "Native glyph request requires current OCR Work and detector caps");
    D3D11_BUFFER_DESC work_desc {};
    ComPtr<ID3D11Device> owner;
    request.work->GetDesc(&work_desc);
    request.work->GetDevice(&owner);
    require(work_desc.ByteWidth == work_bytes && owner.Get() == device_.Get() && (work_desc.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS), "Invalid native glyph Work owner");
    band(request.width, request.band_height);
    job_ = request;
    submitted_work_ = request.work;
    source_ = request.source_rgb;
    caps_ = request.detector_caps;
    if (source_) {
      ComPtr<ID3D11Resource> resource;
      ComPtr<ID3D11Texture2D> texture;
      D3D11_TEXTURE2D_DESC tex {};
      D3D11_SHADER_RESOURCE_VIEW_DESC view {};
      source_->GetResource(&resource);
      source_->GetDesc(&view);
      resource->GetDevice(owner.ReleaseAndGetAddressOf());
      if (SUCCEEDED(resource.As(&texture))) {
        texture->GetDesc(&tex);
      }
      if (owner.Get() != device_.Get() || view.Format != DXGI_FORMAT_R16G16B16A16_FLOAT || view.ViewDimension != D3D11_SRV_DIMENSION_TEXTURE2D || view.Texture2D.MostDetailedMip || tex.Width != width_ || tex.Height != height_ || tex.ArraySize != 1 || tex.MipLevels != 1 || tex.SampleDesc.Count != 1) {
        source_.Reset();
      }
    }
    history_valid_ = history_valid_ && !request.reset_history && same_geometry(request, prior_job_);
    slot_ = 0;
    tile_start_ = 0;
    has_lines_ = false;
    const compute_state_t<16, 5> saved(ctx_.Get());
    ctx_->CopyResource(work_.buffer.Get(), submitted_work_.Get());
    const std::array<UINT, 4> zero {};
    ctx_->ClearUnorderedAccessViewUint(protection_uav_.Get(), zero.data());
    auto &old = history_[previous_];
    dispatch(begin_history, schedule(), 1, 1, {{0, caps_.Get()}, {1, old.lines.srv.Get()}}, {work_.uav.Get(), next().lines.uav.Get(), sizes_.uav.Get()});
    dispatch(confirm_history, schedule(), 8, 1, {{1, old.lines.srv.Get()}, {2, old_rgb_srv_.Get()}, {3, source_.Get()}, {4, old.raw.srv.Get()}}, {nullptr, next().lines.uav.Get()});
    dispatch(finish_windows, schedule(), 1, 1, {}, {work_.uav.Get(), next().lines.uav.Get(), sizes_.uav.Get()});
    ctx_->CopyResource(staged_sizes_.Get(), sizes_.buffer.Get());
    phase_ = phase_e::plan;
    fence();
  }

  poll_status_e poll(const request_t *current, result_t &result) {
    if (phase_ == phase_e::idle || phase_ == phase_e::leased) {
      return poll_status_e::idle;
    }
    const HRESULT query = ctx_->GetData(event_.Get(), nullptr, 0, D3D11_ASYNC_GETDATA_DONOTFLUSH);
    if (query == S_FALSE) {
      return poll_status_e::pending;
    }
    require(query == S_OK, "Native glyph D3D completion failed");
    // Every issued CUDA operation is followed by an unmap, a D3D logits consumer and this fence.
    cuda_outstanding_ = false;
    if (!current || *current != job_) {
      history_valid_ = false;
      drop();
      return poll_status_e::discarded;
    }
    if (phase_ == phase_e::finishing) {
      return publish(result);
    }
    if (phase_ == phase_e::plan) {
      D3D11_MAPPED_SUBRESOURCE map {};
      const HRESULT status = ctx_->Map(staged_sizes_.Get(), 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &map);
      if (status == DXGI_ERROR_WAS_STILL_DRAWING) {
        return poll_status_e::pending;
      }
      require(SUCCEEDED(status), "Read native glyph dispatch extents");
      std::memcpy(sizes_cpu_.data(), map.pData, sizeof(sizes_cpu_));
      ctx_->Unmap(staged_sizes_.Get(), 0);
      UINT pixels = 0;
      for (const auto &[w, h] : sizes_cpu_) {
        require(w <= width_ && h <= height_ && (w == 0) == (h == 0), "Invalid native glyph dispatch extent");
        pixels += w * h;
      }
      // The first fence already covers the cleared mask, empty line records and Work.
      if (pixels == 0) {
        return publish(result);
      }
      ensure(next().raw, pixels, 4);
      has_lines_ = true;
    }
    score_chunk();
    return poll_status_e::pending;
  }

  void retire() {
    if (phase_ != phase_e::leased) {
      return;
    }
    prior_job_ = job_;
    history_valid_ = has_lines_;
    previous_ = 1u - previous_;
    drop();
  }

  bool release() noexcept {
    if (busy() || cuda_outstanding_) {
      return false;
    }
    if (!primary_) {
      return true;
    }
    try {
      const current_cuda_context_t guard(cuda_, primary_);
      for (auto &resource : interop_) {
        if (resource && cuda_.cuGraphicsUnregisterResource(resource) != CUDA_SUCCESS) {
          return false;
        }
        resource = nullptr;
      }
      if (stream_ && cuda_.cuStreamDestroy(stream_) != CUDA_SUCCESS) {
        return false;
      }
      stream_ = nullptr;
    } catch (...) {
      return false;
    }
    if (cuda_.cuDevicePrimaryCtxRelease(adapter_) != CUDA_SUCCESS) {
      return false;
    }
    primary_ = nullptr;
    return true;
  }
};
