// PRIVATE class-body include. Only included by video_depth_estimator::impl.
// RAW OCR availability only: no depth publication, OCR8, current UI or EMPTY authority.
static_assert(raw_ocr_probe_width == static_cast<std::uint32_t>(ocr_engine_width) && raw_ocr_probe_height == static_cast<std::uint32_t>(ocr_engine_height));
bool independent_ocr_raw_enabled = false;
bool independent_ocr_raw_prepared = false;
bool independent_ocr_raw_pending = false;
bool independent_ocr_raw_leased = false;
bool independent_ocr_raw_mapped = false;
bool independent_depth_release_deferred = false;
bool independent_depth_events_valid = false;
std::uint64_t independent_depth_event_sequence = 0u;
CUstream independent_ocr_raw_stream = nullptr;
CUevent independent_ocr_raw_release_event = nullptr;
CUevent independent_depth_compute_event = nullptr;
CUevent independent_depth_hold_event = nullptr;
CUgraphicsResource independent_ocr_raw_mailbox_registration = nullptr;
cuda_conditional_graph::executable_t independent_ocr_raw_wrapper;
detail::cuda_graph_signature_t independent_ocr_raw_signature {};
CUdeviceptr independent_ocr_raw_mailbox_pointer = 0u;
std::uint32_t independent_ocr_raw_sequence = 0u;
std::uint64_t independent_ocr_raw_last_serial = 0u;
std::uint64_t independent_ocr_raw_last_clock = 0u;
raw_ocr_probe_source_t independent_ocr_raw_source {};
Microsoft::WRL::ComPtr<ID3D11Texture2D> independent_ocr_raw_source_copy;
Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> independent_ocr_raw_source_srv;
Microsoft::WRL::ComPtr<ID3D11Texture2D> independent_ocr_raw_band;
Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> independent_ocr_raw_band_srv;
Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> independent_ocr_raw_band_uav;
Microsoft::WRL::ComPtr<ID3D11Buffer> independent_ocr_raw_mailbox;
Microsoft::WRL::ComPtr<ID3D11Buffer> independent_ocr_raw_preprocess_cb;
Microsoft::WRL::ComPtr<ID3D11Buffer> independent_ocr_raw_identity_cb;
D3D11_TEXTURE2D_DESC independent_ocr_raw_source_desc {};
std::array<std::uint32_t, 16u> independent_ocr_raw_identity {};

bool independent_raw_context_ok() const {
  if (!context || context->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE) {
    return false;
  }
  Microsoft::WRL::ComPtr<ID3D11Predicate> predicate;
  BOOL predicate_value = FALSE;
  context->GetPredication(&predicate, &predicate_value);
  return !predicate;
}

bool independent_raw_fail(std::string &error, const char *message, bool execution_uncertain = false) {
  error = message;
  if (execution_uncertain) {
    mark_shared_execution_failure(true);
  }
  return false;
}

// Called only while this private instance exclusively owns the resident OCR context.
// The depth wrapper must never embed ocr_inference_graph in this mode.
bool independent_raw_bind_and_capture(cuda_driver_api &cuda, CUdeviceptr input, CUdeviceptr output, CUdeviceptr mailbox, bool bootstrap, std::string &error) {
  const detail::cuda_graph_signature_t signature {input, output, ocr_engine_width, ocr_engine_height};
  if (!bootstrap) {
    return (signature == independent_ocr_raw_signature &&
            mailbox == independent_ocr_raw_mailbox_pointer && independent_ocr_raw_wrapper.ready()) ||
           independent_raw_fail(error, "OCR remap signature changed: private proof unavailable");
  }
  if (has_previous_frame || independent_ocr_raw_wrapper.ready() || ocr_inference_graph.graph) {
    return independent_raw_fail(error, "OCR bootstrap requires empty isolated graph ownership");
  }
  std::lock_guard<std::mutex> lock(g_trt_mutex);
  if (!ocr_exec_context->setInputShape("x", make_input_dims(ocr_engine_height, ocr_engine_width)) || !ocr_exec_context->setTensorAddress("x", reinterpret_cast<void *>(input)) || !ocr_exec_context->setTensorAddress("fetch_name_0", reinterpret_cast<void *>(output))) {
    return independent_raw_fail(error, "Private OCR shape/address binding failed", true);
  }
  const auto bytes = ocr_exec_context->getMaxOutputSize("fetch_name_0");
  if (bytes <= 0 || static_cast<std::uint64_t>(bytes) > raw_ocr_probe_probability_bytes) {
    return independent_raw_fail(error, "Private OCR output allocation contract failed");
  }
  if (!ocr_exec_context->enqueueV3(independent_ocr_raw_stream)) {
    return independent_raw_fail(error, "Private OCR signature warmup enqueue failed", true);
  }
  joined_stream_ocr_ever_submitted_or_armed = true;
  if (cuda.cuStreamSynchronize(independent_ocr_raw_stream) != CUDA_SUCCESS) {
    return independent_raw_fail(error, "Private OCR signature warmup failed", true);
  }
  CUgraph captured = nullptr;
  const auto begin = cuda.cuStreamBeginCapture(independent_ocr_raw_stream, CU_STREAM_CAPTURE_MODE_RELAXED);
  if (begin != CUDA_SUCCESS) {
    return independent_raw_fail(error, "Private OCR capture begin failed", true);
  }
  const bool enqueued = ocr_exec_context->enqueueV3(independent_ocr_raw_stream);
  const auto end = cuda.cuStreamEndCapture(independent_ocr_raw_stream, &captured);
  ocr_inference_graph.graph = captured;  // Retained/quarantined by existing source-graph teardown.
  if (!enqueued || end != CUDA_SUCCESS || !captured) {
    return independent_raw_fail(error, "Private OCR source capture failed", true);
  }
  cuda_conditional_graph::build_desc_t desc;
  desc.context = cuda_ctx;
  desc.infer_child = captured;
  desc.decision_record = mailbox;
  desc.request_record = mailbox + sizeof(cuda_conditional_graph::decision_record_t);
  auto wrapper = cuda_conditional_graph::executable_t::build(cuda, desc);
  if (!wrapper.ready() || !independent_ocr_raw_wrapper.adopt_from_empty(std::move(wrapper))) {
    return independent_raw_fail(error, "Private OCR mandatory-child wrapper failed", true);
  }
  independent_ocr_raw_signature = signature;
  independent_ocr_raw_mailbox_pointer = mailbox;
  return true;
}

bool independent_raw_submit(ID3D11ShaderResourceView *source, const raw_ocr_probe_source_t &identity, bool bootstrap, std::string &error) {
  if (!independent_ocr_raw_enabled || !is_operational() || !independent_raw_context_ok() || !source || !identity.capture_epoch || !identity.capture_serial || !identity.capture_timestamp_us || identity.source_width != independent_ocr_raw_source_desc.Width || identity.source_height != independent_ocr_raw_source_desc.Height || identity.analysis_generation != independent_ocr_raw_source.analysis_generation || identity.capture_epoch != independent_ocr_raw_source.capture_epoch || independent_ocr_raw_pending || independent_ocr_raw_leased || independent_ocr_raw_mapped || (!bootstrap && (!independent_ocr_raw_prepared || identity.capture_serial <= independent_ocr_raw_last_serial || identity.capture_timestamp_us <= independent_ocr_raw_last_clock))) {
    return independent_raw_fail(error, "Private raw OCR admission/identity/lease rejected");
  }
  auto &cuda = cuda_driver_api::get();
  if (cuda.cuCtxSetCurrent(cuda_ctx) != CUDA_SUCCESS) {
    return independent_raw_fail(error, "Private OCR CUDA context selection failed", true);
  }
  Microsoft::WRL::ComPtr<ID3D11Resource> resource;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
  source->GetResource(&resource);
  D3D11_TEXTURE2D_DESC d {};
  if (FAILED(resource.As(&texture))) {
    return independent_raw_fail(error, "Private OCR source is not Texture2D");
  }
  texture->GetDesc(&d);
  D3D11_SHADER_RESOURCE_VIEW_DESC view {};
  source->GetDesc(&view);
  if (d.Width != identity.source_width || d.Height != identity.source_height || d.Format != DXGI_FORMAT_B8G8R8A8_UNORM || d.MipLevels != 1 || d.ArraySize != 1 || d.SampleDesc.Count != 1 || view.Format != DXGI_FORMAT_B8G8R8A8_UNORM || view.ViewDimension != D3D11_SRV_DIMENSION_TEXTURE2D || view.Texture2D.MostDetailedMip != 0 || view.Texture2D.MipLevels != 1) {
    return independent_raw_fail(error, "Private proof accepts one native BGRA8 SDR source only");
  }
  Microsoft::WRL::ComPtr<ID3D11Device> source_device;
  resource->GetDevice(&source_device);
  if (source_device.Get() != device.Get()) {
    return independent_raw_fail(error, "Private OCR source device mismatch");
  }
  if (independent_ocr_raw_sequence == 0xffffffffu) {
    return independent_raw_fail(error, "Private OCR token sequence exhausted");
  }
  subtitle_cohort_research_configuration_locked = true;  // First raw input locks all flags too.
  const raw_ocr_probe_token_t token {std::uint64_t {0x4f435250u} << 32u | ++independent_ocr_raw_sequence};
  const std::array<std::uint32_t, 16u> words {
    std::uint32_t(identity.capture_serial),
    std::uint32_t(identity.capture_serial >> 32u),
    std::uint32_t(identity.capture_timestamp_us),
    std::uint32_t(identity.capture_timestamp_us >> 32u),
    std::uint32_t(identity.capture_epoch),
    std::uint32_t(identity.capture_epoch >> 32u),
    std::uint32_t(identity.analysis_generation),
    std::uint32_t(identity.analysis_generation >> 32u),
    identity.source_width,
    identity.source_height,
    raw_ocr_probe_width,
    raw_ocr_probe_height,
    std::uint32_t(token.value),
    std::uint32_t(token.value >> 32u),
    0u,
    0u
  };
  const auto crop = subtitle_ocr_source_crop_height(identity.source_width, identity.source_height);
  const std::array<std::uint32_t, 8u> preprocess {
    identity.source_width,
    identity.source_height,
    identity.source_height - crop,
    crop,
    static_cast<std::uint32_t>(input_color_space::srgb),
    0u,
    0u,
    0u
  };

  struct cs_restore_t {
    ID3D11DeviceContext *c;
    ID3D11ComputeShader *shader = nullptr;
    std::array<ID3D11ClassInstance *, 256u> instances {};
    UINT count = 256u;
    ID3D11ShaderResourceView *srv = nullptr;
    ID3D11UnorderedAccessView *uavs[2] {};
    ID3D11Buffer *cb = nullptr;

    explicit cs_restore_t(ID3D11DeviceContext *context):
        c(context) {
      c->CSGetShader(&shader, instances.data(), &count);
      c->CSGetShaderResources(0u, 1u, &srv);
      c->CSGetUnorderedAccessViews(0u, 2u, uavs);
      c->CSGetConstantBuffers(0u, 1u, &cb);
    }

    ~cs_restore_t() {
      ID3D11ShaderResourceView *null_srv = nullptr;
      ID3D11UnorderedAccessView *null_uavs[2] {};
      c->CSSetShaderResources(0u, 1u, &null_srv);
      c->CSSetUnorderedAccessViews(0u, 2u, null_uavs, nullptr);
      c->CSSetShader(shader, instances.data(), count);
      c->CSSetConstantBuffers(0u, 1u, &cb);
      c->CSSetShaderResources(0u, 1u, &srv);
      c->CSSetUnorderedAccessViews(0u, 2u, uavs, nullptr);
      if (shader) {
        shader->Release();
      }
      for (UINT i = 0; i < count; ++i) {
        if (instances[i]) {
          instances[i]->Release();
        }
      }
      if (srv) {
        srv->Release();
      }
      for (auto *u : uavs) {
        if (u) {
          u->Release();
        }
      }
      if (cb) {
        cb->Release();
      }
    }
  } restore(context.Get());

  context->CopyResource(independent_ocr_raw_source_copy.Get(), resource.Get());
  context->UpdateSubresource(independent_ocr_raw_preprocess_cb.Get(), 0, nullptr, preprocess.data(), 0, 0);
  context->UpdateSubresource(independent_ocr_raw_identity_cb.Get(), 0, nullptr, words.data(), 0, 0);
  const auto proposal = cuda_conditional_graph::make_proposal(cuda_conditional_graph::branch_e::infer, token.value);
  const auto request = cuda_conditional_graph::make_request(token.value, cuda_conditional_graph::work_flag_e::none);
  std::array<std::uint32_t, 16u> mailbox {};
  std::memcpy(mailbox.data(), &proposal, sizeof(proposal));
  std::memcpy(mailbox.data() + 8u, &request, sizeof(request));
  context->UpdateSubresource(independent_ocr_raw_mailbox.Get(), 0, nullptr, mailbox.data(), 0, 0);
  context->CSSetShader(ocr_preprocess_cs.Get(), nullptr, 0);
  context->CSSetConstantBuffers(0u, 1u, independent_ocr_raw_preprocess_cb.GetAddressOf());
  context->CSSetShaderResources(0u, 1u, independent_ocr_raw_source_srv.GetAddressOf());
  ID3D11UnorderedAccessView *outputs[2] {ocr_input_uav.Get(), independent_ocr_raw_band_uav.Get()};
  context->CSSetUnorderedAccessViews(0u, 2u, outputs, nullptr);
  context->Dispatch((raw_ocr_probe_width + 15u) / 16u, (raw_ocr_probe_height + 15u) / 16u, 1u);
  ID3D11ShaderResourceView *null_srv = nullptr;
  ID3D11UnorderedAccessView *null_uavs[2] {};
  context->CSSetShaderResources(0u, 1u, &null_srv);
  context->CSSetUnorderedAccessViews(0u, 2u, null_uavs, nullptr);
  CUgraphicsResource resources[3] {cuda_ocr_in_res, cuda_ocr_out_res, independent_ocr_raw_mailbox_registration};
  if (cuda.cuGraphicsMapResources(3u, resources, independent_ocr_raw_stream) != CUDA_SUCCESS) {
    return independent_raw_fail(error, "Private OCR map failed", true);
  }
  independent_ocr_raw_mapped = true;
  CUdeviceptr input = 0, output = 0, control = 0;
  std::size_t input_bytes = 0, output_bytes = 0, control_bytes = 0;
  bool valid = cuda.cuGraphicsResourceGetMappedPointer(&input, &input_bytes, resources[0]) == CUDA_SUCCESS &&
               cuda.cuGraphicsResourceGetMappedPointer(&output, &output_bytes, resources[1]) == CUDA_SUCCESS &&
               cuda.cuGraphicsResourceGetMappedPointer(&control, &control_bytes, resources[2]) == CUDA_SUCCESS &&
               input && output && control && input_bytes >= raw_ocr_probe_probability_bytes * 3u &&
               output_bytes >= raw_ocr_probe_probability_bytes && control_bytes >= 64u;
  if (valid) {
    valid = independent_raw_bind_and_capture(cuda, input, output, control, bootstrap, error);
  }
  if (valid) {
    valid = cuda.cuGraphLaunch(independent_ocr_raw_wrapper.get(), independent_ocr_raw_stream) == CUDA_SUCCESS;
    joined_stream_ocr_ever_submitted_or_armed = true;
  }
  const auto unmap = cuda.cuGraphicsUnmapResources(3u, resources, independent_ocr_raw_stream);
  if (unmap == CUDA_SUCCESS) {
    independent_ocr_raw_mapped = false;
  }
  if (unmap != CUDA_SUCCESS) {
    return independent_raw_fail(error, "Private OCR unmap failed", true);
  }
  if (!valid) {
    independent_ocr_raw_prepared = false;
    if (error.empty()) {
      return independent_raw_fail(error, "Private OCR pointer/launch failed", true);
    }
    return false;  // Exact remap-signature mismatch is unavailable, not a model result.
  }
  if (cuda.cuEventRecord(independent_ocr_raw_release_event, independent_ocr_raw_stream) != CUDA_SUCCESS) {
    return independent_raw_fail(error, "Private OCR release event record failed", true);
  }
  independent_ocr_raw_source = identity;
  independent_ocr_raw_identity = words;
  independent_ocr_raw_last_serial = identity.capture_serial;
  independent_ocr_raw_last_clock = identity.capture_timestamp_us;
  independent_ocr_raw_pending = true;
  return true;
}

bool independent_raw_prepare(ID3D11ShaderResourceView *warm_source, const raw_ocr_probe_source_t &identity, std::string &error) {
  auto &cuda = cuda_driver_api::get();
  if (!independent_ocr_raw_enabled || independent_ocr_raw_prepared || has_previous_frame || !is_operational() || !independent_raw_context_ok() || !warm_source || !identity.capture_epoch || !identity.capture_serial || !identity.capture_timestamp_us || !identity.source_width || !identity.source_height || identity.source_width > 5120u || identity.source_height > 5120u || !ocr_available || !ocr_exec_context || !ocr_preprocess_cs || !cuda_ocr_in_res || !cuda_ocr_out_res || !cuda.cuGraphInstantiateWithFlags || !cuda.cuStreamBeginCapture || !cuda.cuStreamEndCapture || !cuda.cuStreamSynchronize || !cuda.cuEventRecord || !cuda.cuEventQuery || cuda.cuCtxSetCurrent(cuda_ctx) != CUDA_SUCCESS) {
    return independent_raw_fail(error, "Private OCR prepare precondition unavailable");
  }
  independent_ocr_raw_source = identity;
  auto &d = independent_ocr_raw_source_desc;
  d.Width = identity.source_width;
  d.Height = identity.source_height;
  d.MipLevels = d.ArraySize = d.SampleDesc.Count = 1u;
  d.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
  d.Usage = D3D11_USAGE_DEFAULT;
  d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
  bool ok = SUCCEEDED(device->CreateTexture2D(&d, nullptr, &independent_ocr_raw_source_copy)) &&
            SUCCEEDED(device->CreateShaderResourceView(independent_ocr_raw_source_copy.Get(), nullptr, &independent_ocr_raw_source_srv));
  auto band_desc = d;
  band_desc.Height = subtitle_ocr_source_crop_height(d.Width, d.Height);
  band_desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
  band_desc.BindFlags |= D3D11_BIND_UNORDERED_ACCESS;
  const auto create_cb = [&](UINT bytes, Microsoft::WRL::ComPtr<ID3D11Buffer> &buffer) {
    D3D11_BUFFER_DESC desc {};
    desc.ByteWidth = bytes;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    return SUCCEEDED(device->CreateBuffer(&desc, nullptr, &buffer));
  };
  ok = ok && SUCCEEDED(device->CreateTexture2D(&band_desc, nullptr, &independent_ocr_raw_band)) &&
       SUCCEEDED(device->CreateShaderResourceView(independent_ocr_raw_band.Get(), nullptr, &independent_ocr_raw_band_srv)) &&
       SUCCEEDED(device->CreateUnorderedAccessView(independent_ocr_raw_band.Get(), nullptr, &independent_ocr_raw_band_uav)) &&
       create_cb(32u, independent_ocr_raw_preprocess_cb) &&
       create_cb(64u, independent_ocr_raw_identity_cb);
  D3D11_BUFFER_DESC control {};
  control.ByteWidth = 64u;
  control.Usage = D3D11_USAGE_DEFAULT;
  control.BindFlags = D3D11_BIND_SHADER_RESOURCE;
  control.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
  control.StructureByteStride = 4u;
  ok = ok && SUCCEEDED(device->CreateBuffer(&control, nullptr, &independent_ocr_raw_mailbox));
  if (!ok) {
    return independent_raw_fail(error, "Private OCR D3D allocation failed");
  }
  if (cuda.cuStreamCreate(&independent_ocr_raw_stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS || cuda.cuEventCreate(&independent_ocr_raw_release_event, CU_EVENT_DEFAULT) != CUDA_SUCCESS || cuda.cuGraphicsD3D11RegisterResource(&independent_ocr_raw_mailbox_registration, independent_ocr_raw_mailbox.Get(), 0u) != CUDA_SUCCESS) {
    return independent_raw_fail(error, "Private OCR CUDA allocation/registration failed", true);
  }
  if (!independent_raw_submit(warm_source, identity, true, error)) {
    return false;
  }
  if (cuda.cuStreamSynchronize(independent_ocr_raw_stream) != CUDA_SUCCESS) {
    return independent_raw_fail(error, "Private OCR prepared inference failed", true);
  }
  // Caller must read the genuine prepared output/receipt and release its contents lease.
  independent_ocr_raw_prepared = true;
  return true;
}

raw_ocr_probe_poll_t independent_raw_poll() {
  raw_ocr_probe_poll_t result;
  if (!independent_ocr_raw_enabled || !independent_ocr_raw_prepared || !independent_ocr_raw_pending || is_terminal()) {
    return result;
  }
  auto &cuda = cuda_driver_api::get();
  if (cuda.cuCtxSetCurrent(cuda_ctx) != CUDA_SUCCESS) {
    mark_shared_execution_failure(true);
    return result;
  }
  const auto status = cuda.cuEventQuery(independent_ocr_raw_release_event);
  result.actual_release_query = static_cast<std::int32_t>(status);
  if (status == CUDA_ERROR_NOT_READY) {
    result.pending = true;
    return result;
  }
  if (status != CUDA_SUCCESS) {
    mark_shared_execution_failure(true);
    return result;
  }
  independent_ocr_raw_leased = true;
  raw_ocr_probe_views_t views;
  views.token = {std::uint64_t {0x4f435250u} << 32u | independent_ocr_raw_sequence};
  views.source = independent_ocr_raw_source;
  views.current_source = independent_ocr_raw_source_srv;
  views.source_band = independent_ocr_raw_band_srv;
  views.probability = ocr_output_srv;
  views.transaction = independent_ocr_raw_mailbox;
  views.source_identity_constants = independent_ocr_raw_identity_cb;
  views.preprocess_constants = independent_ocr_raw_preprocess_cb;
  result.views = std::move(views);
  return result;
}

bool independent_raw_release(raw_ocr_probe_token_t token) {
  if (!independent_ocr_raw_leased || !independent_ocr_raw_pending || token.value != (std::uint64_t {0x4f435250u} << 32u | independent_ocr_raw_sequence)) {
    return false;
  }
  // Caller promises all D3D readers/copies of this lease have completed before releasing it.
  independent_ocr_raw_leased = independent_ocr_raw_pending = false;
  return true;
}

CUresult independent_depth_advance_release(cuda_driver_api &cuda, bool wait) {
  if (!independent_depth_release_deferred) {
    return CUDA_SUCCESS;
  }
  if (!independent_depth_events_valid) {
    return static_cast<CUresult>(-1);
  }
  if (wait && cuda.cuEventSynchronize(independent_depth_hold_event) != CUDA_SUCCESS) {
    return static_cast<CUresult>(-1);
  }
  const auto ready = cuda.cuEventQuery(independent_depth_hold_event);
  if (benchmark_joined_completion_gate) {
    benchmark_joined_completion_gate->note_actual_event_query(pending_frame_id, ready);
  }
  if (ready != CUDA_SUCCESS) {
    return ready;
  }
  CUgraphicsResource depth[3] {cuda_in_res, cuda_out_res, cuda_near_identical_decision_res};
  if (benchmark_joined_completion_gate) {
    benchmark_joined_completion_gate->note_unmap_begin(pending_frame_id, joined_completion_gate_benchmark::unmap_e::depth);
  }
  const auto unmapped = cuda.cuGraphicsUnmapResources(3u, depth, cu_stream);
  if (benchmark_joined_completion_gate) {
    benchmark_joined_completion_gate->note_unmap_return(pending_frame_id, joined_completion_gate_benchmark::unmap_e::depth, unmapped);
  }
  if (unmapped != CUDA_SUCCESS) {
    return unmapped;
  }
  independent_depth_release_deferred = false;
  return record_inference_done_event(cuda, depth_inference_done_event, cu_stream, pending_depth_inference_event_recorded, "private deferred depth release", false) ? CUDA_SUCCESS : static_cast<CUresult>(-1);
}

bool independent_raw_quiesce(cuda_driver_api &cuda) {
  if (!independent_ocr_raw_enabled) {
    return true;
  }
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds {250};
  if (independent_ocr_raw_stream) {
    while (true) {
      const auto q = cuda.cuStreamQuery(independent_ocr_raw_stream);
      if (q == CUDA_SUCCESS) {
        break;
      }
      if (q != CUDA_ERROR_NOT_READY || std::chrono::steady_clock::now() >= deadline) {
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds {1});
    }
  }
  if (independent_depth_release_deferred && independent_depth_advance_release(cuda, false) != CUDA_SUCCESS) {
    return false;
  }
  return !independent_ocr_raw_mapped && (!cu_stream || cuda.cuStreamQuery(cu_stream) == CUDA_SUCCESS);
}

void independent_raw_abandon() noexcept {
  independent_ocr_raw_wrapper.abandon_unsafe();
  (void) independent_ocr_raw_mailbox.Detach();
  (void) independent_ocr_raw_source_copy.Detach();
  (void) independent_ocr_raw_source_srv.Detach();
  (void) independent_ocr_raw_band.Detach();
  (void) independent_ocr_raw_band_srv.Detach();
  (void) independent_ocr_raw_band_uav.Detach();
  (void) independent_ocr_raw_preprocess_cb.Detach();
  (void) independent_ocr_raw_identity_cb.Detach();
  // Raw CUDA handles intentionally retained, never reused after an uncertain error.
}

bool independent_raw_cleanup(cuda_driver_api &cuda) {
  if (!independent_ocr_raw_enabled) {
    return true;
  }
  if (!independent_ocr_raw_wrapper.reset()) {
    return false;
  }
  if (independent_ocr_raw_mailbox_registration) {
    if (cuda.cuGraphicsUnregisterResource(independent_ocr_raw_mailbox_registration) != CUDA_SUCCESS) {
      return false;
    }
    independent_ocr_raw_mailbox_registration = nullptr;
  }
  for (CUevent *e : {&independent_ocr_raw_release_event, &independent_depth_compute_event, &independent_depth_hold_event}) {
    if (*e && cuda.cuEventDestroy(*e) != CUDA_SUCCESS) {
      return false;
    }
    *e = nullptr;
  }
  if (independent_ocr_raw_stream && cuda.cuStreamDestroy(independent_ocr_raw_stream) != CUDA_SUCCESS) {
    return false;
  }
  independent_ocr_raw_stream = nullptr;
  independent_ocr_raw_pending = independent_ocr_raw_leased = false;
  independent_ocr_raw_prepared = false;
  return true;
}
