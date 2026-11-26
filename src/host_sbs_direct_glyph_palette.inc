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
    field_t lines, opacity;
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
  ComPtr<ID3D11DeviceContext4> ctx_;
  std::array<ComPtr<ID3D11ComputeShader>, stage_count> shaders_;
  buffer_ptr constants_, staged_sizes_;
  ComPtr<ID3D11Fence> completion_;
  UINT64 submitted_fence_value_ = 0;
  field_t work_, sizes_, input_, logits_, roles_, low_, fine_, peak_, mean_;
  // Views of the shared encoder tensors that begin at each tile of the batch.
  std::array<uav_ptr, tile_capacity> input_tiles_;
  std::array<srv_ptr, tile_capacity> logits_tiles_;
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
  struct schedule_readback_t {
    std::array<std::array<UINT, 2>, 8> lines;
    UINT hold_history, reserved;
    std::array<std::array<UINT, 2>, 8> reference;
  };
  static_assert(sizeof(schedule_readback_t) == 136);
  schedule_readback_t plan_ {};
  UINT slot_ = 0, tile_start_ = 0, encoder_width_ = 0, detail_width_ = 0, step_ = 1, hist_scale_ = 0;
  UINT encoder_rows_ = 28, detail_rows_ = 84;
  bool has_lines_ = false;
  cuda_driver_api &cuda_ = cuda_driver_api::get();
  CUcontext primary_ = nullptr;
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
           a.subtitle_context_generation == b.subtitle_context_generation &&
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

  void dispatch(stage_e stage, const std::array<UINT, 8> &cb, UINT gx, UINT gy, std::initializer_list<binding_t> inputs, targets_t outputs, UINT gz = 1) {
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
    ctx_->Dispatch(gx, gy, gz);
    reads.fill(nullptr);
    outputs.fill(nullptr);
    ctx_->CSSetShaderResources(0, UINT(reads.size()), reads.data());
    ctx_->CSSetUnorderedAccessViews(0, UINT(outputs.size()), outputs.data(), nullptr);
  }

  std::array<UINT, 8> schedule() const {
    const UINT source_flags = UINT(source_ != nullptr) | (job_.video_caption_ocr ? 2u : 0u);
    return {width_, height_, UINT(job_.token), UINT(job_.token >> 32u), source_flags, UINT(history_valid_), slot_, job_.line_center_twice_source};
  }

  std::array<UINT, 8> line_constants(UINT arg) const {
    return {slot_, encoder_width_, detail_width_, arg, hist_scale_, 0, 0, encoder_rows_};
  }

  void stage(stage_e entry, UINT arg, UINT gx, UINT gy, std::initializer_list<binding_t> inputs, field_t &destination, field_t *second = nullptr) {
    dispatch(entry, line_constants(arg), gx, gy, inputs, {destination.uav.Get(), second ? second->uav.Get() : nullptr});
  }

  void fence() {
    require(SUCCEEDED(ctx_->Signal(completion_.Get(), ++submitted_fence_value_)), "Signal native glyph completion");
    // The file worker may have no further D3D commands to submit until the scheduler wakes.
    ctx_->Flush();
  }

  void encode(UINT tiles) {
    constexpr UINT count = 2;
    require(execution_->setInputShape("rgb", nvinfer1::Dims4 {int(tiles), 3, int(palette_tile.height), int(palette_tile.width)}), "Select native glyph tile batch");
    cuda_outstanding_ = true;
    check_cuda(cuda_.cuGraphicsMapResources(count, interop_.data(), stream_), "Map native glyph encoder bindings");
    std::array<CUdeviceptr, count> addr {};
    for (UINT i = 0; i < count; ++i) {
      std::size_t bytes = 0;
      check_cuda(cuda_.cuGraphicsResourceGetMappedPointer(&addr[i], &bytes, interop_[i]), "Map native glyph tensor");
      require(addr[i] && bytes == sizeof(float) * (i == 0 ? palette_input_words : palette_output_words), "Native glyph encoder tensor extent mismatch");
    }
    require(execution_->setTensorAddress("rgb", reinterpret_cast<void *>(addr[0])) && execution_->setTensorAddress("logits", reinterpret_cast<void *>(addr[1])), "Bind mixed native glyph encoder");
    require(execution_->enqueueV3(stream_), "Submit mixed native glyph tile batch");
    check_cuda(cuda_.cuGraphicsUnmapResources(count, interop_.data(), stream_), "Return native glyph tensors to D3D");
  }

  // The OCR work height samples 28 encoder rows. A taller detector cover adds rows up to the tile's.
  UINT encoder_rows(UINT slot) const {
    const UINT h = plan_.lines[slot][1], reference_height = plan_.reference[slot][0];
    require(reference_height && reference_height <= h, "Invalid native glyph scale reference");
    return std::clamp((28u * h + reference_height / 2u) / reference_height, 28u, palette_tile.height);
  }

  UINT planned_tiles() const {
    UINT tiles = 0;
    for (UINT slot = 0; slot < plan_.lines.size(); ++slot) {
      tiles += line_tiles(slot);
    }
    return tiles;
  }

  UINT line_tiles(UINT slot) const {
    const auto [w, h] = plan_.lines[slot];
    return w ? div_up(div_up(w * encoder_rows(slot), h), 440u) : 0u;
  }

  void select_line(UINT slot) {
    slot_ = slot;
    const auto [w, h] = plan_.lines[slot];
    encoder_rows_ = encoder_rows(slot);
    detail_rows_ = 3u * encoder_rows_;
    encoder_width_ = div_up(w * encoder_rows_, h);
    detail_width_ = div_up(w * detail_rows_, h);
    // Match the reference's ties-to-even integer sampling step.
    step_ = std::max(1u, UINT(std::nearbyint(double(h) / 96.0)));
    hist_scale_ = std::min(65536u, 0x7fffffffu / (div_up(w, step_) * div_up(h, step_)));
  }

  // Sizes the selected line's scratch and samples its encoder lattice.
  void begin_line() {
    const auto [w, h] = plan_.lines[slot_];
    const UINT n = detail_width_ * detail_rows_;
    ensure(low_, encoder_width_ * encoder_rows_, 4);
    ensure(roles_, encoder_width_ * encoder_rows_, 16);
    ensure(peak_, encoder_width_ * encoder_rows_, 16);
    ensure(mean_, encoder_width_ * encoder_rows_, 16);
    ensure(fine_, n, 4);
    for (auto *field : {&moment_, &squared_, &h1_, &h2_, &near_, &far_}) {
      ensure(*field, n, 16);
    }
    ensure(parts_, 2u * div_up(n, 128), 16);
    ensure(alpha_, w * h, 4);
    stage(pack_rgb, encoder_rows_, div_up(encoder_width_, 16), div_up(encoder_rows_, 8), {{0, source_.Get()}}, low_);
  }

  // Packs the selected line's tiles from an encoder column into the batch from batch_tile.
  void pack_tiles(UINT column, UINT batch_tile, UINT tiles) {
    dispatch(pack_tile, line_constants(column), div_up(palette_tile.width, 16), div_up(palette_tile.height, 8), {{7, low_.srv.Get()}}, {input_tiles_[batch_tile].Get()}, tiles);
  }

  // Reads the same tiles' logits back into the selected line's encoder roles.
  void export_tiles(UINT column, UINT batch_tile, UINT tiles) {
    dispatch(export_roles, line_constants(column), div_up(std::min(440u, encoder_width_ - column), 16), div_up(encoder_rows_, 8), {{12, logits_tiles_[batch_tile].Get()}}, {roles_.uav.Get()}, tiles);
  }

  void finish_line() {
    const auto [w, h] = plan_.lines[slot_];
    const std::array<UINT, 4> zero {};
    ctx_->ClearUnorderedAccessViewUint(hist_.uav.Get(), zero.data());
    stage(pack_rgb, detail_rows_, div_up(detail_width_, 16), div_up(detail_rows_, 8), {{0, source_.Get()}}, fine_);
    // Use the same sampled pixel centers for the color counts and their areas.
    const UINT pad = std::max(1u, UINT(std::nearbyint(double(h) / 8.0)));
    const auto interior_extent = [&](UINT length) {
      return length > 2u * pad ? div_up(length - pad, step_) - div_up(pad, step_) : 0u;
    };
    const UINT inside = interior_extent(w) * interior_extent(h);
    const UINT total = div_up(w, step_) * div_up(h, step_);
    stage(coarse, 0, div_up(encoder_width_, 16), div_up(encoder_rows_, 8), {{1, roles_.srv.Get()}}, peak_, &mean_);
    stage(make_hist, pad, div_up(div_up(w, step_), 16), div_up(div_up(h, step_), 8), {{0, source_.Get()}, {1, roles_.srv.Get()}}, hist_);
    stage(smooth_hist, 0, 39, 1, {{9, hist_.srv.Get()}}, lut_);
    stage(smooth_hist, 1, 39, 1, {{10, lut_.srv.Get()}}, temp_);
    stage(smooth_hist, 2, 39, 1, {{10, temp_.srv.Get()}}, lut_, &palette_);
    dispatch(choose_colors, {slot_, encoder_width_, detail_width_, 0, hist_scale_, total - inside, inside, encoder_rows_},
      1, 1, {{1, roles_.srv.Get()}, {4, palette_.srv.Get()}, {10, lut_.srv.Get()}}, {winner_.uav.Get()});
    stage(make_moments, 0, div_up(detail_width_, 16), div_up(detail_rows_, 8), {{1, roles_.srv.Get()}, {8, fine_.srv.Get()}}, moment_, &squared_);
    for (UINT radius : {6u, 17u}) {
      stage(smooth_moments, radius, div_up(detail_width_, 16), div_up(detail_rows_, 8), {{12, moment_.srv.Get()}, {13, squared_.srv.Get()}}, h1_, &h2_);
      stage(smooth_moments, 64u + radius, div_up(detail_width_, 16), div_up(detail_rows_, 8), {{12, h1_.srv.Get()}, {13, h2_.srv.Get()}}, radius == 6u ? near_ : far_);
    }
    const UINT groups = div_up(detail_width_ * detail_rows_, 128);
    stage(prototypes, 0, std::min(groups, 65535u), div_up(groups, 65535u), {{1, roles_.srv.Get()}, {8, fine_.srv.Get()}}, parts_);
    stage(prototypes, 1, 1, 1, {{15, parts_.srv.Get()}}, proto_);
    stage(decode_palette, 0, div_up(w, 16), div_up(h, 8), {{0, source_.Get()}, {1, roles_.srv.Get()}, {2, peak_.srv.Get()}, {3, mean_.srv.Get()}, {4, palette_.srv.Get()}, {5, winner_.srv.Get()}, {6, near_.srv.Get()}, {8, fine_.srv.Get()}, {14, far_.srv.Get()}, {15, proto_.srv.Get()}}, alpha_);
    auto &old = history_[previous_];
    dispatch(confirm_mask_history, schedule(), 1, 1, {{1, old.lines.srv.Get()}, {2, old_rgb_srv_.Get()}, {3, source_.Get()}, {4, old.opacity.srv.Get()}, {5, alpha_.srv.Get()}, {7, winner_.srv.Get()}}, {nullptr, next().lines.uav.Get()});
    dispatch(publish_line, schedule(), div_up(w, 16), div_up(h, 8), {{1, old.lines.srv.Get()}, {2, old_rgb_srv_.Get()}, {3, source_.Get()}, {4, old.opacity.srv.Get()}, {5, alpha_.srv.Get()}, {6, next().lines.srv.Get()}}, {work_.uav.Get(), nullptr, nullptr, next().opacity.uav.Get(), protection_uav_.Get()});
  }

  // Scores every planned line in one encoder batch, each from its own batch position.
  void score_lines() {
    UINT tiles = 0;
    for (UINT slot = 0; slot < plan_.lines.size(); ++slot) {
      if (const UINT count = line_tiles(slot)) {
        select_line(slot);
        begin_line();
        pack_tiles(0, tiles, count);
        tiles += count;
      }
    }
    encode(tiles);
    for (UINT slot = 0, batch_tile = 0; slot < plan_.lines.size(); ++slot) {
      if (const UINT count = line_tiles(slot)) {
        select_line(slot);
        export_tiles(0, batch_tile, count);
        finish_line();
        batch_tile += count;
      }
    }
    slot_ = UINT(plan_.lines.size());
  }

  void score_chunk() {
    const current_cuda_context_t guard(cuda_, primary_);
    const palette_state_t saved(ctx_.Get());
    // When every planned tile fits one batch, the lines share it and leave no chunk below.
    if (phase_ == phase_e::plan && planned_tiles() <= tile_capacity) {
      score_lines();
    }
    while (slot_ < plan_.lines.size()) {
      if (!plan_.lines[slot_][0]) {
        ++slot_;
        continue;
      }
      if (tile_start_ == 0) {
        select_line(slot_);
        begin_line();
      }
      const UINT remaining = encoder_width_ - tile_start_;
      // Balance the tiles across the fewest submissions to avoid a small final batch.
      const UINT remaining_tiles = div_up(remaining, 440u);
      const UINT batches = div_up(remaining_tiles, tile_capacity);
      const UINT tiles = div_up(remaining_tiles, batches);
      pack_tiles(tile_start_, 0, tiles);
      encode(tiles);
      export_tiles(tile_start_, 0, tiles);
      tile_start_ += tiles * 440u;
      if (tile_start_ >= encoder_width_) {
        finish_line();
        ++slot_;
        tile_start_ = 0;
      }
      // Retire this batch's GPU fence before changing the execution context's shape.
      break;
    }
    while (slot_ < plan_.lines.size() && plan_.lines[slot_][0] == 0) {
      ++slot_;
    }
    if (slot_ == plan_.lines.size()) {
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
  // CUDA's D3D11 registration reaches the device's immediate context, so it runs on the thread
  // that draws with that context (early as depth starts, else at the first submission), never
  // in prepare() on the background build worker: registering there, beside the drawing thread,
  // deadlocked inside NVIDIA's driver.
  void register_interop() {
    if (interop_[0] && interop_[1]) {
      return;
    }
    const current_cuda_context_t guard(cuda_, primary_);
    if (!interop_[0]) {
      check_cuda(cuda_.cuGraphicsD3D11RegisterResource(&interop_[0], input_.buffer.Get(), 0), "Register native glyph RGB");
    }
    if (!interop_[1]) {
      check_cuda(cuda_.cuGraphicsD3D11RegisterResource(&interop_[1], logits_.buffer.Get(), 0), "Register native glyph logits");
    }
  }

  void prepare(ID3D11Device *device, ID3D11DeviceContext *context, const std::filesystem::path &root, CUcontext primary, nvinfer1::IExecutionContext *execution) {
    require(device && context && primary && execution, "Native glyph transport requires an execution lease");
    device_ = device;
    ComPtr<ID3D11Device5> device5;
    require(SUCCEEDED(device_.As(&device5)) && SUCCEEDED(context->QueryInterface(IID_PPV_ARGS(&ctx_))) &&
            SUCCEEDED(device5->CreateFence(0, D3D11_FENCE_FLAG_NONE, IID_PPV_ARGS(&completion_))),
            "Create native glyph D3D11 completion fence");
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
    require(execution_->setInputShape("rgb", nvinfer1::Dims4 {int(tile_capacity), 3, int(palette_tile.height), int(palette_tile.width)}) &&
            execution_->getMaxOutputSize("logits") == sizeof(float) * palette_output_words, "Native glyph requires the bounded mixed tile batch");
    require(cuda_.is_valid(), "Native glyph CUDA interop is unavailable");
    // The caller's TensorRT lease runs on this primary context, which the caller keeps retained
    // for as long as this transport may use it.
    primary_ = primary;
    const current_cuda_context_t guard(cuda_, primary_);
    check_cuda(cuda_.cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING), "Create bounded native glyph stream");
    ensure(input_, palette_input_words, 4);
    ensure(logits_, palette_output_words, 4);
    for (UINT tile = 0; tile < tile_capacity; ++tile) {
      D3D11_UNORDERED_ACCESS_VIEW_DESC rgb {};
      rgb.Format = DXGI_FORMAT_UNKNOWN;
      rgb.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
      rgb.Buffer.FirstElement = tile * palette_tile.input_words();
      rgb.Buffer.NumElements = (tile_capacity - tile) * palette_tile.input_words();
      D3D11_SHADER_RESOURCE_VIEW_DESC scores {};
      scores.Format = DXGI_FORMAT_UNKNOWN;
      scores.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
      scores.Buffer.FirstElement = tile * palette_tile.output_words();
      scores.Buffer.NumElements = (tile_capacity - tile) * palette_tile.output_words();
      require(SUCCEEDED(device->CreateUnorderedAccessView(input_.buffer.Get(), &rgb, &input_tiles_[tile])) && SUCCEEDED(device->CreateShaderResourceView(logits_.buffer.Get(), &scores, &logits_tiles_[tile])), "Create native glyph batch tile views");
    }
    constants_ = make_buffer({32, D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0});
    staged_sizes_ = make_buffer({sizeof(schedule_readback_t), D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ, 0, 0});
    work_.buffer = make_buffer({work_bytes, D3D11_USAGE_DEFAULT, D3D11_BIND_UNORDERED_ACCESS, 0, D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS, 0});
    D3D11_UNORDERED_ACCESS_VIEW_DESC v {};
    v.Format = DXGI_FORMAT_R32_TYPELESS;
    v.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    v.Buffer.NumElements = work_bytes / 4u;
    v.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
    require(SUCCEEDED(device->CreateUnorderedAccessView(work_.buffer.Get(), &v, &work_.uav)), "Create native glyph Work view");
    ensure(sizes_, sizeof(schedule_readback_t) / (2 * sizeof(UINT)), 2 * sizeof(UINT));
    for (auto &generation : history_) {
      ensure(generation.lines, 8, 64);
      ensure(generation.opacity, 1, 2 * sizeof(float));
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

#ifdef SUNSHINE_TESTS
  UINT64 submitted_fences_for_tests() const noexcept {
    return submitted_fence_value_;
  }

  bool interop_registered_for_tests() const noexcept {
    return interop_[0] && interop_[1];
  }
#endif

  void submit(const request_t &request) {
    require(!busy() && request.work && request.detector_caps && request.token && request.width && request.width <= 5120u && request.band_height && request.band_height <= 5120u, "Native glyph request requires current OCR Work and detector caps");
    register_interop();
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
    const palette_state_t saved(ctx_.Get());
    ctx_->CopyResource(work_.buffer.Get(), submitted_work_.Get());
    const std::array<UINT, 4> zero {};
    ctx_->ClearUnorderedAccessViewUint(protection_uav_.Get(), zero.data());
    auto &old = history_[previous_];
    dispatch(begin_history, schedule(), 1, 1, {{0, caps_.Get()}, {1, old.lines.srv.Get()}, {8, request.video_caption_ocr}}, {work_.uav.Get(), next().lines.uav.Get(), sizes_.uav.Get()});
    dispatch(confirm_history, schedule(), 8, 1, {{1, old.lines.srv.Get()}, {2, old_rgb_srv_.Get()}, {3, source_.Get()}, {4, old.opacity.srv.Get()}}, {nullptr, next().lines.uav.Get()});
    dispatch(finish_windows, schedule(), 1, 1, {}, {work_.uav.Get(), next().lines.uav.Get(), sizes_.uav.Get()});
    ctx_->CopyResource(staged_sizes_.Get(), sizes_.buffer.Get());
    phase_ = phase_e::plan;
    fence();
  }

  poll_status_e poll(const request_t *current, result_t &result) {
    if (phase_ == phase_e::idle || phase_ == phase_e::leased) {
      return poll_status_e::idle;
    }
    // This check also runs on the teardown worker, which cannot use the immediate context.
    const UINT64 completed = completion_->GetCompletedValue();
    require(completed != UINT64_MAX, "Native glyph D3D device removed");
    if (completed < submitted_fence_value_) {
      return poll_status_e::pending;
    }
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
      std::memcpy(&plan_, map.pData, sizeof(plan_));
      ctx_->Unmap(staged_sizes_.Get(), 0);
      UINT pixels = 0;
      for (const auto &[w, h] : plan_.lines) {
        require(w <= width_ && h <= height_ && (w == 0) == (h == 0), "Invalid native glyph dispatch extent");
        pixels += w * h;
      }
      // The first fence already covers the cleared mask, empty line records and Work.
      if (pixels == 0) {
        return publish(result);
      }
      ensure(next().opacity, pixels, 2 * sizeof(float));
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
    if (!plan_.hold_history || !history_valid_) {
      history_valid_ = has_lines_;
      previous_ = 1u - previous_;
    }
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
    primary_ = nullptr;
    return true;
  }
};
