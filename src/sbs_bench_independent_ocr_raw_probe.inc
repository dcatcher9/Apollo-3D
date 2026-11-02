// Included beside the existing jp_* private benchmark helpers, after opts is defined.
// No renderer, positive glyph permission, currentEMPTY or live-capture claim.
int run_independent_ocr_raw_probe(const opts &o, const std::vector<fs::path> &frames, const std::vector<std::uint64_t> &timestamps, ID3D11Device *dev, ID3D11DeviceContext *ctx, models::video_depth_estimator &est, const std::string &plan_bytes) {
  namespace util = external_pre_unmap_feasibility_v1;
  namespace gate = models::joined_completion_gate_benchmark;
  constexpr std::size_t count = 6u;
  std::array<const char *, count> names {{"probability.f32", "source-band.rgba16f", "current-source.bgra8", "transaction.u32", "source-identity.u32", "preprocess.u32"}};

  // Observation-only fixed scalar aggregate: no strings/vectors/JSON/GPU handles.
  enum class c_attempt_stage_t : std::uint32_t {
    not_started = 0u,
    depth_B_submit = 1u,
    wait_B_gate = 2u,
    B_before_query = 3u,
    C_source_identity = 4u,
    C_submit = 5u,
    C_release_poll = 6u,
    C_resource_bind = 7u,
    C_GPU_copy = 8u,
    C_D3D_ready = 9u,
    C_owned_Map = 10u,
    B_after_query = 11u,
    early_C_gate_satisfied = 12u
  };

  struct c_attempt_t {
    c_attempt_stage_t stage = c_attempt_stage_t::not_started;
    c_attempt_stage_t failure_stage = c_attempt_stage_t::not_started;
    bool inner_caught = false, outer_caught = false;
    bool before_B_issued = false, after_B_issued = false;
    bool source_assigned = false, submit_called = false, submit_returned = false, submit_result = false;
    models::raw_ocr_probe_source_t source {};
    models::raw_ocr_probe_depth_events_t before_B {}, after_B {};
    std::uint64_t poll_started = 0u, poll_returned = 0u, query0_count = 0u, query600_count = 0u, other_query_count = 0u;
    std::int32_t first_query = -1, last_query = -1;
    std::int64_t first_poll_begin_us = 0, first_poll_return_us = 0, last_poll_begin_us = 0, last_poll_return_us = 0;
    bool first_pending = false, last_pending = false, first_has_views = false, last_has_views = false;
    bool guard_release_flag_observed = false, guard_release_flag = false;
    std::uint64_t guard_release_flag_poll = 0u;
    std::int64_t caught_us = 0;
    std::int64_t submit_begin_us = 0, submit_return_us = 0, raw_ready_us = 0, copy_begin_us = 0;
    std::int64_t graphics_ready_us = 0, map_done_us = 0, before_query_us = 0, after_query_us = 0;
    HRESULT graphics_status = S_FALSE;
  } C_attempt;

  const auto attempt_json = [&]() {
    const auto event = [](const models::raw_ocr_probe_depth_events_t &e) {
      return nlohmann::json {{"deferred_mapped_depth_lease", e.deferred_mapped_depth_lease}, {"actual_compute_query", e.actual_compute_query}, {"actual_hold_query", e.actual_hold_query}, {"depth_frame_id", e.depth_frame_id}, {"depth_observation_timestamp_us", e.depth_observation_timestamp_us}, {"recorded_event_sequence", e.recorded_event_sequence}};
    };
    return nlohmann::json {{"schema", 1u}, {"stage", static_cast<std::uint32_t>(C_attempt.stage)}, {"failure_stage", static_cast<std::uint32_t>(C_attempt.failure_stage)}, {"inner_caught", C_attempt.inner_caught}, {"outer_caught", C_attempt.outer_caught}, {"before_B_query_issued", C_attempt.before_B_issued}, {"after_B_query_issued", C_attempt.after_B_issued}, {"before_B", event(C_attempt.before_B)}, {"after_B", event(C_attempt.after_B)}, {"source_assigned", C_attempt.source_assigned}, {"source", {{"capture_epoch", C_attempt.source.capture_epoch}, {"capture_serial", C_attempt.source.capture_serial}, {"capture_timestamp_us", C_attempt.source.capture_timestamp_us}, {"analysis_generation", C_attempt.source.analysis_generation}, {"source_width", C_attempt.source.source_width}, {"source_height", C_attempt.source.source_height}}}, {"submit_called", C_attempt.submit_called}, {"submit_returned", C_attempt.submit_returned}, {"submit_result", C_attempt.submit_result}, {"poll_started", C_attempt.poll_started}, {"poll_returned", C_attempt.poll_returned}, {"first_actual_release_query", C_attempt.first_query}, {"last_actual_release_query", C_attempt.last_query}, {"query0_count", C_attempt.query0_count}, {"query600_count", C_attempt.query600_count}, {"other_query_count", C_attempt.other_query_count}, {"first_poll_begin_us", C_attempt.first_poll_begin_us}, {"first_poll_return_us", C_attempt.first_poll_return_us}, {"last_poll_begin_us", C_attempt.last_poll_begin_us}, {"last_poll_return_us", C_attempt.last_poll_return_us}, {"first_pending", C_attempt.first_pending}, {"last_pending", C_attempt.last_pending}, {"first_has_views", C_attempt.first_has_views}, {"last_has_views", C_attempt.last_has_views}, {"guard_release_flag_observed", C_attempt.guard_release_flag_observed}, {"guard_release_flag", C_attempt.guard_release_flag}, {"guard_release_flag_poll", C_attempt.guard_release_flag_poll}, {"caught_us", C_attempt.caught_us}, {"existing_timings", {{"before_query_us", C_attempt.before_query_us}, {"submit_begin_us", C_attempt.submit_begin_us}, {"submit_return_us", C_attempt.submit_return_us}, {"raw_ready_us", C_attempt.raw_ready_us}, {"D3D_copy_begin_us", C_attempt.copy_begin_us}, {"D3D_event_ready_us", C_attempt.graphics_ready_us}, {"C_only_Map_owned_copy_done_us", C_attempt.map_done_us}, {"after_query_us", C_attempt.after_query_us}, {"graphics_GetData", C_attempt.graphics_status}}}, {"scalar_record_is_not_extra_GPU_query_or_completion_authority", true}};
  };

  struct capture_t {
    std::array<std::vector<char>, count> data;
    std::array<bool, count> copied {};
    nlohmann::json metadata;
  };

  std::array<jp_stage_t, count> stages;
  std::array<capture_t, 3u> captures;
  std::array<const char *, 3u> phases {{"prepared_A", "current_C_while_B_held", "current_C_after_B_ACK"}};
  const auto root = fs::path(o.out) / "independent-ocr-raw-proof-v1";
  nlohmann::json diagnostic = nlohmann::json::object();
  bool created = false, flushed = false;
  const auto flush = [&]() {
    if (flushed) {
      return;
    }
    flushed = true;
    for (std::size_t p = 0; p < captures.size(); ++p) {
      nlohmann::json artifacts = nlohmann::json::object();
      const auto folder = root / phases[p];
      bool any = false;
      for (std::size_t k = 0; k < count; ++k) {
        if (captures[p].copied[k]) {
          if (!any) {
            fs::create_directories(folder);
            any = true;
          }
          const auto file = folder / names[k];
          std::ofstream out(file, std::ios::binary);
          out.write(captures[p].data[k].data(), static_cast<std::streamsize>(captures[p].data[k].size()));
          out.flush();
          if (!out.good()) {
            throw std::runtime_error("Raw OCR proof file write failed");
          }
          artifacts[names[k]] = jp_file(file);
        }
      }
      if (any) {
        auto j = captures[p].metadata;
        j["artifacts"] = artifacts;
        jp_json(folder / "phase.json", j);
      }
    }
  };
  try {
    const auto plan = nlohmann::json::parse(plan_bytes);
    const bool independent = plan.at("run_kind") == "independent";
    if (plan.at("schema") != 1 || plan.at("probe_kind") != "independent-ocr-raw-v1" || (!independent && plan.at("run_kind") != "ordinary-control") || frames.size() != 3 || timestamps != std::vector<std::uint64_t> {33334u, 66667u, 100000u} || plan.at("hold_after_entry_ms") != 120 || plan.at("callback_safety_timeout_ms") != 500 || fs::exists(root)) {
      throw std::runtime_error("Invalid raw OCR proof plan or existing output");
    }
    fs::create_directories(root);
    created = true;
    std::array<jp_texture_t, 3u> source;
    std::array<ID3D11ShaderResourceView *, 3u> input {};
    nlohmann::json bindings = nlohmann::json::array();
    UINT width = 0, height = 0;
    for (std::size_t i = 0; i < 3; ++i) {
      const auto &b = plan.at("sources").at(i);
      std::string bytes;
      rgba_image image;
      if (fs::weakly_canonical(b.at("path").get<std::string>()) != fs::weakly_canonical(frames[i]) || !read_file_snapshot(frames[i], bytes) || sha256_hex(bytes) != b.at("sha256").get<std::string>() || bytes.size() != b.at("bytes").get<std::size_t>() || !load_png(std::string_view(bytes), image)) {
        throw std::runtime_error("Raw OCR source binding failed");
      }
      if (!i) {
        width = image.w;
        height = image.h;
      }
      if (width != image.w || height != image.h || !width || !height || width > 5120u || height > 5120u) {
        throw std::runtime_error("Raw OCR source extent changed");
      }
      source[i] = jp_texture(dev, width, height, DXGI_FORMAT_B8G8R8A8_UNORM, D3D11_BIND_SHADER_RESOURCE, image.bgra.data(), width * 4u);
      input[i] = source[i].srv.Get();
      auto actual = b;
      actual["decoded_bgra8_sha256"] = sha256_hex(std::string_view(reinterpret_cast<const char *>(image.bgra.data()), image.bgra.size()));
      bindings.push_back(actual);
    }
    const auto resource = [](ID3D11ShaderResourceView *srv) {
      ComPtr<ID3D11Resource> r;
      if (srv) {
        srv->GetResource(&r);
      }
      return r;
    };
    const auto prepare = [&](std::size_t k, ID3D11Resource *r) {
      auto &s = stages[k];
      r->GetType(&s.dimension);
      if (s.dimension == D3D11_RESOURCE_DIMENSION_BUFFER) {
        ComPtr<ID3D11Buffer> b;
        jp_require(r->QueryInterface(IID_PPV_ARGS(&b)), "Raw stage buffer QI");
        D3D11_BUFFER_DESC d {};
        b->GetDesc(&d);
        s.buffer_bytes = s.row_bytes = d.ByteWidth;
        s.rows = 1;
        d.Usage = D3D11_USAGE_STAGING;
        d.BindFlags = d.MiscFlags = d.StructureByteStride = 0;
        d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ComPtr<ID3D11Buffer> stage;
        jp_require(dev->CreateBuffer(&d, nullptr, &stage), "Raw buffer staging");
        jp_require(stage.As(&s.resource), "Raw buffer staging QI");
      } else if (s.dimension == D3D11_RESOURCE_DIMENSION_TEXTURE2D) {
        ComPtr<ID3D11Texture2D> t, stage;
        jp_require(r->QueryInterface(IID_PPV_ARGS(&t)), "Raw texture QI");
        t->GetDesc(&s.texture);
        if (s.texture.Format != DXGI_FORMAT_B8G8R8A8_UNORM && s.texture.Format != DXGI_FORMAT_R16G16B16A16_FLOAT) {
          throw std::runtime_error("Raw texture format unsupported");
        }
        s.row_bytes = std::size_t(s.texture.Width) * (s.texture.Format == DXGI_FORMAT_R16G16B16A16_FLOAT ? 8u : 4u);
        s.rows = s.texture.Height;
        auto d = s.texture;
        d.Usage = D3D11_USAGE_STAGING;
        d.BindFlags = d.MiscFlags = 0;
        d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        jp_require(dev->CreateTexture2D(&d, nullptr, &stage), "Raw texture staging");
        jp_require(stage.As(&s.resource), "Raw texture staging QI");
      } else {
        throw std::runtime_error("Raw staging resource kind unsupported");
      }
      for (auto &c : captures) {
        c.data[k].resize(s.row_bytes * s.rows);
      }
    };
    const auto copy = [&](std::size_t p, std::size_t k, ID3D11Resource *r) {
      auto &s = stages[k];
      if (!r || !s.resource || captures[p].copied[k]) {
        throw std::runtime_error("Raw copy contract");
      }
      D3D11_RESOURCE_DIMENSION kind {};
      r->GetType(&kind);
      if (kind != s.dimension) {
        throw std::runtime_error("Raw copy shape changed");
      }
      ctx->CopyResource(s.resource.Get(), r);
    };
    const auto map_owned = [&](std::size_t p, std::size_t k) {
      auto &s = stages[k];
      auto &c = captures[p];
      D3D11_MAPPED_SUBRESOURCE mapped {};
      jp_require(ctx->Map(s.resource.Get(), 0u, D3D11_MAP_READ, 0u, &mapped), "Raw C-only staging Map");
      for (std::size_t y = 0; y < s.rows; ++y) {
        std::memcpy(c.data[k].data() + y * s.row_bytes, static_cast<const char *>(mapped.pData) + y * (s.dimension == D3D11_RESOURCE_DIMENSION_TEXTURE2D ? mapped.RowPitch : s.row_bytes), s.row_bytes);
      }
      ctx->Unmap(s.resource.Get(), 0u);
      c.copied[k] = true;
    };
    const auto submit_depth = [&](std::size_t i, std::uint64_t frame, std::uint64_t observation) {
      const models::gpu_adaptive_reuse_request force {.observation_timestamp_us = observation};
      return est.estimate_depth(input[i], models::input_color_space::srgb, frame, false, {}, models::depth_optional_work_mode_e::ordinary, force, 0);
    };
    const auto finish_depth = [&](std::uint64_t frame, std::uint64_t observation) {
      auto r = est.finish_pending_depth_for_evaluation(models::input_color_space::srgb, false);
      util::require(r.completed_frame_valid && r.completed_frame_id == frame && r.publication.observation_timestamp_us == observation && models::parallax_v2_result_is_authenticated(r), "Raw OCR proof depth owner ACK failed");
      return r;
    };
    diagnostic = {{"schema", 1}, {"probe_kind", "independent-ocr-raw-v1"}, {"run_kind", plan.at("run_kind")}, {"plan_sha256", sha256_hex(plan_bytes)}, {"sources", bindings}, {"logical_P", {960, 160}}, {"raw_availability_only", true}, {"semantic_status", "UNKNOWN"}, {"performance_qualified", false}, {"no_current_UI_or_EMPTY_or_renderer_authority", true}};
    util::require(est.set_completed_ocr_conditioner_research_enabled(true), "Completed control view flag failed");
    if (independent) {
      util::require(est.set_independent_ocr_raw_probe_enabled(true), "Raw private pre-input flag failed");
      gate::configuration_t c;
      c.enabled = true;
      c.placement = gate::placement_e::before_interop_unmaps;
      c.target_frame_id = 3;
      c.hold_after_callback_enter_ms = 120;
      c.callback_entry_timeout_ms = 30000;
      c.callback_safety_timeout_ms = 500;
      util::require(est.configure_joined_completion_gate_for_benchmark(c), "Raw depth hold setup failed");
    }

    struct release_t {
      models::video_depth_estimator &est;

      ~release_t() {
        est.release_joined_completion_gate_for_benchmark();
      }
    } release {est};

    util::require(submit_depth(2, 1, 1).inference_enqueued, "Raw empty depth warmup enqueue failed");
    auto warm = finish_depth(1, 1);
    util::require(submit_depth(0, 2, 33334).inference_enqueued, "Raw A depth enqueue failed");
    auto A = finish_depth(2, 33334);
    diagnostic["warmup_publication"] = jp_publication(warm.publication);
    diagnostic["A_publication"] = jp_publication(A.publication);
    util::require(A.raw_model_provenance && A.composite_depth_runtime_provenance, "Actual model provenance unavailable");
    const auto &raw_model = *A.raw_model_provenance;
    const auto &composite_model = *A.composite_depth_runtime_provenance;
    diagnostic["raw_model"] = {{"model", raw_model.depth_model}, {"onnx_sha256", raw_model.onnx_sha256}, {"preprocess_profile", raw_model.preprocess_profile}, {"preprocess_source_closure_sha256", raw_model.preprocess_source_closure_sha256}};
    diagnostic["composite_model"] = {{"model", composite_model.model}, {"onnx_sha256", composite_model.onnx_sha256}, {"embedded_dav2_onnx_sha256", composite_model.embedded_dav2_onnx_sha256}, {"zipdepth_checkpoint_sha256", composite_model.zipdepth_checkpoint_sha256}, {"guidance_preprocess_source_closure_sha256", composite_model.guidance_preprocess_source_closure_sha256}, {"engine_recipe", composite_model.engine_recipe}, {"engine_artifact", composite_model.engine_artifact}, {"active_engine_manifest", composite_model.active_engine_manifest}};
    if (!independent) {
      phases = {{"ordinary_B", "ordinary_C", "unused"}};
      names[4] = "control-contour-CB.u32";
      names[5] = "control-arguments.u32";
      for (std::size_t p = 0; p < 2u; ++p) {
        const auto input_index = p + 1u;
        const std::uint64_t frame = p + 3u;
        const std::uint64_t observation = p == 0 ? 66667u : 100000u;
        util::require(submit_depth(input_index, frame, observation).inference_enqueued, "Ordinary control OCR enqueue failed");
        auto owner = finish_depth(frame, observation);
        const auto views = est.completed_ocr_conditioner_working_views_for_evaluation(owner.publication);
        util::require(views.has_value(), "Ordinary same-current OCR snapshot unavailable");
        auto P = resource(views->probability_snapshot.Get()), band = resource(views->source_band.Get());
        if (!p) {
          prepare(0, P.Get());
          prepare(1, band.Get());
          prepare(2, source[0].texture.Get());
          prepare(3, views->transaction.Get());
          prepare(4, views->constants.Get());
          prepare(5, views->dispatch_arguments.Get());
        }
        copy(p, 0, P.Get());
        copy(p, 1, band.Get());
        copy(p, 2, source[input_index].texture.Get());
        copy(p, 3, views->transaction.Get());
        copy(p, 4, views->constants.Get());
        copy(p, 5, views->dispatch_arguments.Get());
        for (std::size_t k = 0; k < count; ++k) {
          map_owned(p, k);
        }
        captures[p].metadata = {{"publication", jp_publication(owner.publication)}, {"input_index", input_index}, {"source_own_model_frame", frame}, {"source_own_model_observation_us", observation}};
      }
      flush();
      jp_json(root / "report.json", diagnostic);
      return 0;
    }
    std::string error;
    error.reserve(256u);
    const auto capture_epoch = static_cast<std::uint64_t>(util::now_us());
    models::raw_ocr_probe_source_t raw_source {capture_epoch, 1u, static_cast<std::uint64_t>(util::now_us()), 0u, width, height};
    util::require(est.prepare_independent_ocr_raw_probe(input[0], raw_source, error), error.c_str());
    auto prepared = est.poll_independent_ocr_raw_probe();
    util::require(prepared.actual_release_query == CUDA_SUCCESS && prepared.views.has_value(), "Prepared raw A genuine release unavailable");
    const auto &pv = *prepared.views;
    std::array<ComPtr<ID3D11Resource>, count> raw_resources {{resource(pv.probability.Get()), resource(pv.source_band.Get()), resource(pv.current_source.Get()), nullptr, nullptr, nullptr}};
    jp_require(pv.transaction.As(&raw_resources[3]), "Raw mailbox resource QI");
    jp_require(pv.source_identity_constants.As(&raw_resources[4]), "Raw identity CB resource QI");
    jp_require(pv.preprocess_constants.As(&raw_resources[5]), "Raw preprocess CB resource QI");
    for (std::size_t k = 0; k < count; ++k) {
      prepare(k, raw_resources[k].Get());
      copy(0, k, raw_resources[k].Get());
      map_owned(0, k);
    }
    captures[0].metadata = {{"bootstrap_only", true}, {"source_capture_serial", raw_source.capture_serial}, {"source_capture_timestamp_us", raw_source.capture_timestamp_us}, {"capture_epoch", capture_epoch}, {"token", pv.token.value}, {"release_event_query", prepared.actual_release_query}};
    util::require(est.release_independent_ocr_raw_probe(pv.token), "Prepared A contents lease release failed");
    D3D11_QUERY_DESC query_desc {D3D11_QUERY_EVENT, 0u};
    ComPtr<ID3D11Query> copied;
    jp_require(dev->CreateQuery(&query_desc, &copied), "Raw C completion query allocation");
    // All uploads, graphics/CUDA allocations, JIT/captures and CPU vectors precede B hold.
    models::raw_ocr_probe_depth_events_t before {}, after {};
    models::raw_ocr_probe_poll_t current;
    // Aliases retain the existing timestamp sites/values, with fixed storage surviving failure.
    auto &submit_begin = C_attempt.submit_begin_us;
    auto &submit_return = C_attempt.submit_return_us;
    auto &raw_ready = C_attempt.raw_ready_us;
    auto &copy_begin = C_attempt.copy_begin_us;
    auto &graphics_ready = C_attempt.graphics_ready_us;
    auto &map_done = C_attempt.map_done_us;
    auto &before_query = C_attempt.before_query_us;
    auto &after_query = C_attempt.after_query_us;
    auto &graphics_status = C_attempt.graphics_status;
    const char *availability = "UNAVAILABLE";
    try {
      C_attempt.stage = c_attempt_stage_t::depth_B_submit;
      util::require(submit_depth(1, 3, 66667).inference_enqueued, "Actual B depth submit failed");
      C_attempt.stage = c_attempt_stage_t::wait_B_gate;
      const auto wait_limit = std::chrono::steady_clock::now() + std::chrono::seconds {30};
      while (!util::status(est).callback_entered && std::chrono::steady_clock::now() < wait_limit) {
        std::this_thread::yield();
      }
      C_attempt.stage = c_attempt_stage_t::B_before_query;
      before_query = util::now_us();
      C_attempt.before_B_issued = true;
      before = est.independent_ocr_raw_depth_events_for_evaluation();
      C_attempt.before_B = before;
      util::require(before.deferred_mapped_depth_lease && before.actual_hold_query == CUDA_ERROR_NOT_READY && !util::status(est).released && before.depth_frame_id == 3u && before.depth_observation_timestamp_us == 66667u && before.recorded_event_sequence == 3u, "No genuine fresh B hold event NOT_READY window");
      C_attempt.stage = c_attempt_stage_t::C_source_identity;
      raw_source.capture_serial = 3u;
      raw_source.capture_timestamp_us = static_cast<std::uint64_t>(util::now_us());
      C_attempt.source = raw_source;
      C_attempt.source_assigned = true;
      C_attempt.stage = c_attempt_stage_t::C_submit;
      C_attempt.submit_called = true;
      submit_begin = util::now_us();
      const bool submitted = est.submit_independent_ocr_raw_probe(input[2], raw_source, error);
      submit_return = util::now_us();
      C_attempt.submit_returned = true;
      C_attempt.submit_result = submitted;
      util::require(submitted, error.c_str());
      C_attempt.stage = c_attempt_stage_t::C_release_poll;
      while (true) {
        ++C_attempt.poll_started;
        C_attempt.last_poll_begin_us = util::now_us();
        if (C_attempt.poll_started == 1u) {
          C_attempt.first_poll_begin_us = C_attempt.last_poll_begin_us;
        }
        current = est.poll_independent_ocr_raw_probe();
        C_attempt.last_poll_return_us = util::now_us();
        ++C_attempt.poll_returned;
        C_attempt.last_query = current.actual_release_query;
        C_attempt.last_pending = current.pending;
        C_attempt.last_has_views = current.views.has_value();
        if (C_attempt.poll_returned == 1u) {
          C_attempt.first_poll_return_us = C_attempt.last_poll_return_us;
          C_attempt.first_query = current.actual_release_query;
          C_attempt.first_pending = current.pending;
          C_attempt.first_has_views = current.views.has_value();
        }
        if (current.actual_release_query == CUDA_SUCCESS) {
          ++C_attempt.query0_count;
        } else if (current.actual_release_query == CUDA_ERROR_NOT_READY) {
          ++C_attempt.query600_count;
        } else {
          ++C_attempt.other_query_count;
        }
        if (current.views) {
          break;
        }
        // Preserve the original short-circuit: exactly one EXISTING CPU gate snapshot
        // only if pending. No additional CUDA/D3D query is introduced.
        bool existing_guard_ok = false;
        if (current.pending) {
          C_attempt.guard_release_flag_observed = true;
          C_attempt.guard_release_flag = util::status(est).released;
          C_attempt.guard_release_flag_poll = C_attempt.poll_returned;
          existing_guard_ok = !C_attempt.guard_release_flag;
        }
        util::require(current.pending && existing_guard_ok, "Raw C did not release before B hold release");
        std::this_thread::yield();
      }
      raw_ready = util::now_us();
      C_attempt.stage = c_attempt_stage_t::C_resource_bind;
      const auto &cv = *current.views;
      util::require(cv.token.value != pv.token.value && cv.source.capture_serial == 3u && cv.source.capture_timestamp_us == raw_source.capture_timestamp_us, "Raw C source/token changed");
      // Same fixed allocations; no B tensor/transaction resource is copied or Map-read here.
      raw_resources[0] = resource(cv.probability.Get());
      raw_resources[1] = resource(cv.source_band.Get());
      raw_resources[2] = resource(cv.current_source.Get());
      jp_require(cv.transaction.As(&raw_resources[3]), "Raw C mailbox QI");
      jp_require(cv.source_identity_constants.As(&raw_resources[4]), "Raw C identity QI");
      jp_require(cv.preprocess_constants.As(&raw_resources[5]), "Raw C preprocess QI");
      C_attempt.stage = c_attempt_stage_t::C_GPU_copy;
      copy_begin = util::now_us();
      for (std::size_t k = 0; k < count; ++k) {
        copy(1, k, raw_resources[k].Get());
      }
      C_attempt.stage = c_attempt_stage_t::C_D3D_ready;
      ctx->End(copied.Get());
      ctx->Flush();
      while ((graphics_status = ctx->GetData(copied.Get(), nullptr, 0u, D3D11_ASYNC_GETDATA_DONOTFLUSH)) == S_FALSE) {
        util::require(!util::status(est).released, "Raw C D3D did not complete before B gate release");
        std::this_thread::yield();
      }
      jp_require(graphics_status, "Raw C D3D completion query failed");
      graphics_ready = util::now_us();
      C_attempt.stage = c_attempt_stage_t::C_owned_Map;
      for (std::size_t k = 0; k < count; ++k) {
        map_owned(1, k);
      }
      map_done = util::now_us();
      C_attempt.stage = c_attempt_stage_t::B_after_query;
      after_query = util::now_us();
      C_attempt.after_B_issued = true;
      after = est.independent_ocr_raw_depth_events_for_evaluation();
      C_attempt.after_B = after;
      util::require(after.deferred_mapped_depth_lease && after.actual_hold_query == CUDA_ERROR_NOT_READY && !util::status(est).released && after.depth_frame_id == 3u && after.depth_observation_timestamp_us == 66667u && after.recorded_event_sequence == before.recorded_event_sequence, "Raw C readback did not finish inside actual B NOT_READY");
      C_attempt.stage = c_attempt_stage_t::early_C_gate_satisfied;
      availability = "RAW_C_GRAPH_RELEASE_D3D_AND_OWNED_CPU_COPY_COMPLETED_WHILE_B_HELD";
    } catch (const std::exception &e) {
      C_attempt.inner_caught = true;
      C_attempt.failure_stage = C_attempt.stage;
      C_attempt.caught_us = util::now_us();
      error = e.what();
    }
    // Independent OS release remains effective even if any CUDA/D3D call above stalled.
    est.release_joined_completion_gate_for_benchmark();
    models::estimate_result B;
    if (!est.has_terminal_failure()) {
      B = finish_depth(3u, 66667u);
    }
    const auto final_gate = util::status(est);
    if (current.views && availability[0] != 'U' && !est.has_terminal_failure()) {
      // Current-C lease remains held across late B ACK. No new source observation is invented.
      for (std::size_t k = 0; k < count; ++k) {
        copy(2, k, raw_resources[k].Get());
        map_owned(2, k);
      }
      for (std::size_t k = 0; k < count; ++k) {
        util::require(captures[1].data[k] == captures[2].data[k], "Late B ACK overwrote current C raw contents");
      }
      captures[2].metadata = {{"same_C_contents_lease_after_B_ACK", true}, {"token", current.views->token.value}};
    }
    if (current.views) {
      (void) est.release_independent_ocr_raw_probe(current.views->token);
    }
    // No hash/file/JSON work before the release and B drain above.
    captures[1].metadata = {{"source_capture_serial", raw_source.capture_serial}, {"source_capture_timestamp_us", raw_source.capture_timestamp_us}, {"capture_epoch", capture_epoch}, {"token", current.views ? current.views->token.value : 0u}, {"raw_release_query", current.actual_release_query}, {"before_B_hold_query", before.actual_hold_query}, {"after_B_hold_query", after.actual_hold_query}, {"before_B_compute_query", before.actual_compute_query}, {"after_B_compute_query", after.actual_compute_query}, {"before_query_us", before_query}, {"submit_begin_us", submit_begin}, {"submit_return_us", submit_return}, {"raw_ready_us", raw_ready}, {"D3D_copy_begin_us", copy_begin}, {"D3D_event_ready_us", graphics_ready}, {"C_only_Map_owned_copy_done_us", map_done}, {"after_query_us", after_query}, {"graphics_GetData", graphics_status}};
    captures[1].metadata["held_depth_frame_id"] = before.depth_frame_id;
    captures[1].metadata["held_depth_observation_timestamp_us"] = before.depth_observation_timestamp_us;
    captures[1].metadata["before_recorded_event_sequence"] = before.recorded_event_sequence;
    captures[1].metadata["after_recorded_event_sequence"] = after.recorded_event_sequence;
    // Unconditional attempt publication only AFTER release/drain, even zero C copies.
    diagnostic["C_attempt"] = attempt_json();
    diagnostic["availability"] = availability;
    diagnostic["unavailable_reason"] = error;
    diagnostic["gate"] = jp_gate(final_gate);
    diagnostic["B_publication"] = jp_publication(B.publication);
    diagnostic["no_B_resource_readback_while_held"] = true;
    diagnostic["hashes_computed_only_after_release_and_B_drain"] = true;
    diagnostic["pre_gate_compute_can_already_be_ready"] = true;
    flush();
    jp_json(root / "report.json", diagnostic);
    return availability[0] == 'U' || est.has_terminal_failure() ? 12 : 0;
  } catch (const std::exception &e) {
    est.release_joined_completion_gate_for_benchmark();
    if (!est.has_terminal_failure()) {
      (void) est.finish_pending_depth_for_evaluation(models::input_color_space::srgb, false);
    }
    if (created) {
      flush();
      C_attempt.outer_caught = true;
      diagnostic["C_attempt"] = attempt_json();
      diagnostic["fatal_error"] = e.what();
      jp_json(root / "failure.json", diagnostic);
    }
    return 12;
  }
}
