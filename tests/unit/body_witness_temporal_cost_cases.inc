// PRIVATE default-off functional/cost fixture. Included inside existing anonymous namespace.
// Every original source/model observation remains literal. No renderer/model/scorer is run.
class HostSbsBodyWitnessTemporalCostGpuTest: public HostSbsBodyWitnessTemporalGpuTest {
protected:
  using clock_t = std::chrono::steady_clock;
  using base_t = models::host_sbs_subtitle_inner_colour_gpu_global_validation_v2::resources_t;

  struct sample_t {
    nlohmann::json item;
    std::vector<char> rgb, p, s, ocr, cb, frame, root, base_body, base_matte, temporal_body, temporal_matte;
    UINT components = 0u;
    bool measured = false;
  };

  struct capture_t {
    ComPtr<ID3D11Resource> source;
    colour_archive_t stage;
  };

  struct call_t {
    std::map<std::string, capture_t> captures;
    ComPtr<ID3D11Query> start, end;
    UINT64 cpu_ns = 0u;
    std::uint64_t recordings = 0u, replays = 0u;
  };

  static void need(bool ok, const std::string &why) {
    if (!ok) {
      throw std::runtime_error(why);
    }
  }

  static UINT64 elapsed(clock_t::time_point a, clock_t::time_point z) {
    return UINT64(std::chrono::duration_cast<std::chrono::nanoseconds>(z - a).count());
  }

  ComPtr<ID3D11Query> query(D3D11_QUERY kind) {
    D3D11_QUERY_DESC d {};
    d.Query = kind;
    ComPtr<ID3D11Query> q;
    need(SUCCEEDED(device_->CreateQuery(&d, &q)), "Cost query allocation failed");
    return q;
  }

  void wait_query(ID3D11Query *q, void *out, UINT size) {
    const auto deadline = clock_t::now() + std::chrono::seconds(60);
    HRESULT hr;
    while ((hr = context_->GetData(q, out, size, D3D11_ASYNC_GETDATA_DONOTFLUSH)) == S_FALSE) {
      need(clock_t::now() < deadline, "Cost query timeout");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    need(hr == S_OK, "Cost query failure");
  }

  capture_t prepare_capture(ID3D11Resource *source) {
    capture_t c;
    c.source = source;
    ComPtr<ID3D11Texture2D> t;
    ComPtr<ID3D11Buffer> b;
    if (SUCCEEDED(source->QueryInterface(IID_PPV_ARGS(&t)))) {
      D3D11_TEXTURE2D_DESC d {};
      t->GetDesc(&d);
      need(d.ArraySize == 1u && d.MipLevels == 1u, "Cost capture requires literal native2D");
      c.stage.row_bytes = d.Width * (d.Format == DXGI_FORMAT_R16G16B16A16_FLOAT ? 8u : 4u);
      c.stage.height = d.Height;
      d.Usage = D3D11_USAGE_STAGING;
      d.BindFlags = d.MiscFlags = 0u;
      d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      need(SUCCEEDED(device_->CreateTexture2D(&d, nullptr, &c.stage.texture)), "Cost texture staging allocation failed");
    } else {
      need(SUCCEEDED(source->QueryInterface(IID_PPV_ARGS(&b))), "Cost capture unknown resource");
      D3D11_BUFFER_DESC d {};
      b->GetDesc(&d);
      c.stage.row_bytes = d.ByteWidth;
      c.stage.height = 1u;
      d.Usage = D3D11_USAGE_STAGING;
      d.BindFlags = d.MiscFlags = d.StructureByteStride = 0u;
      d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      need(SUCCEEDED(device_->CreateBuffer(&d, nullptr, &c.stage.buffer)), "Cost buffer staging allocation failed");
    }
    return c;
  }

  capture_t prepare_capture(ID3D11ShaderResourceView *view) {
    need(view != nullptr, "Cost missing view");
    ComPtr<ID3D11Resource> r;
    view->GetResource(&r);
    return prepare_capture(r.Get());
  }

  void copy_capture(const capture_t &c) {
    ID3D11Resource *to = c.stage.texture ? static_cast<ID3D11Resource *>(c.stage.texture.Get()) : c.stage.buffer.Get();
    context_->CopyResource(to, c.source.Get());
  }

  nlohmann::json save(const std::filesystem::path &p, const std::vector<char> &bytes) {
    std::filesystem::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary);
    f.write(bytes.data(), std::streamsize(bytes.size()));
    need(bool(f), "Cost output write failed");
    return {{"path", p.string()}, {"sha256", colour_sha(bytes.data(), bytes.size())}, {"bytes", bytes.size()}};
  }

  std::vector<sample_t> load_samples(const nlohmann::json &sequence) {
    std::vector<sample_t> out;
    for (const auto &item : sequence.at("cases")) {
      sample_t s;
      s.item = item;
      const auto &a = item.at("artifacts");
      s.cb = colour_bound(a.at("constants"), 96u);
      const auto cb = colour_words(s.cb);
      const std::size_t n = std::size_t(cb[0]) * cb[3];
      s.rgb = colour_bound(a.at("source_band"), n * 8u);
      s.p = colour_bound(a.at("probability"), 614400u);
      s.s = colour_bound(a.at("current_support"), n * 4u);
      s.ocr = colour_bound(a.at("ocr_record"), 1024u);
      s.frame = colour_bound(a.at("frame_constants"), 256u);
      s.root = colour_bound(a.at("root_arguments"), 12u);
      need(colour_words(s.root) == std::vector<UINT>({1u, 1u, 1u}), "Actual cost observation must use literal ROOT1");
      s.base_body = colour_bound(item.at("expected").at("baseline_body"), n * 4u);
      s.base_matte = colour_bound(item.at("expected").at("baseline_final"), n * 4u);
      s.temporal_body = colour_bound(item.at("expected").at("body"), n * 4u);
      s.temporal_matte = colour_bound(item.at("expected").at("final"), n * 4u);
      const auto raw_details = colour_bound(item.at("provenance").at("candidate_details"), item.at("provenance").at("candidate_details").at("bytes").get<std::size_t>());
      s.components = UINT(nlohmann::json::parse(raw_details.begin(), raw_details.end()).at("decisions").size());
      s.measured = colour_words(s.ocr)[4u] != 0u;
      need(s.measured == (s.components != 0u), "Predeclared20positive/8empty classification disagrees source OCR/component reference");
      out.push_back(std::move(s));
    }
    return out;
  }

  void upload(const sample_t &s) {
    context_->UpdateSubresource(constants_buffer_.Get(), 0u, nullptr, s.cb.data(), 0u, 0u);
    context_->UpdateSubresource(source_texture_.value.Get(), 0u, nullptr, s.rgb.data(), width_ * 8u, 0u);
    context_->UpdateSubresource(support_texture_.value.Get(), 0u, nullptr, s.s.data(), width_ * 4u, 0u);
    context_->UpdateSubresource(p_buffer_.value.Get(), 0u, nullptr, s.p.data(), 0u, 0u);
    context_->UpdateSubresource(ocr_buffer_.value.Get(), 0u, nullptr, s.ocr.data(), 0u, 0u);
    context_->UpdateSubresource(root_arguments_.Get(), 0u, nullptr, s.root.data(), 0u, 0u);
  }

  bool dispatch_cost(bool temporal_enabled, const sample_t &s) {
    std::array<ID3D11ComputeShader *, 5u> colour {};
    std::array<ID3D11ComputeShader *, 6u> graph {};
    for (UINT i = 0u; i < colour.size(); ++i) {
      colour[i] = inner_shaders_[i].Get();
    }
    for (UINT i = 0u; i < graph.size(); ++i) {
      graph[i] = component_shaders_[i].Get();
    }
    if (!temporal_enabled) {
      return cost_baseline_.dispatch(context_.Get(), colour, graph, constants_buffer_.Get(), source_texture_.srv.Get(), p_buffer_.srv.Get(), support_texture_.srv.Get(), ocr_buffer_.srv.Get(), root_arguments_.Get(), 0u, error_);
    }
    models::host_sbs_body_witness_temporal_recorded::program_t program;
    program.resources = &backend_;
    std::memcpy(program.frame.words.data(), s.frame.data(), 256u);
    program.include_baseline_control = false;
    for (UINT i = 0u; i < program.shaders.size(); ++i) {
      program.shaders[i] = temporal_shaders_[i].Get();
    }
    return coordinator_.dispatch(context_.Get(), colour, graph, constants_buffer_.Get(), source_texture_.srv.Get(), p_buffer_.srv.Get(), support_texture_.srv.Get(), ocr_buffer_.srv.Get(), root_arguments_.Get(), 0u, error_, &program);
  }

  std::pair<std::uint64_t, std::uint64_t> counters(bool temporal_enabled) {
    if (temporal_enabled) {
      const auto d = coordinator_.diagnostics();
      need(d.direct_emissions == 0u && d.unavailable == 0u, "Temporal cost path must remain recorded/available");
      return {d.recordings, d.replays};
    }
    const auto d = cost_baseline_.execution_diagnostics();
    need(cost_baseline_.last_execution_path() == base_t::execution_path_t::recorded && d.predication_direct_fallbacks == 0u, "Baseline cost path must remain recorded");
    return {d.recordings, d.replays};
  }

  nlohmann::json run_scope(bool temporal_enabled, const nlohmann::json &sequence, const std::vector<sample_t> &samples, const std::filesystem::path &destination, bool measured, UINT warm_play) {
    need(!samples.empty(), "Empty cost scope");
    context_->ClearState();
    context_->SetPredication(nullptr, FALSE);
    const auto allocation_start = clock_t::now();
    const auto cb = colour_words(samples.front().cb);
    need(resize(cb[0], cb[1], cb[2], cb[3]), "Cost input allocation failed");
    cost_baseline_ = base_t {};
    backend_ = temporal::resources_t {};
    coordinator_ = models::host_sbs_body_witness_temporal_recorded::resources_t {};
    if (temporal_enabled) {
      need(backend_.ensure(device_.Get(), width_, crop_height_, capacities_, error_) && coordinator_.ensure(device_.Get(), width_, crop_height_, error_), error_);
      coordinator_.set_recorded_execution(true);
    } else {
      need(cost_baseline_.ensure(device_.Get(), width_, crop_height_, error_), error_);
      cost_baseline_.set_recorded_execution(true);
    }
    const auto allocation_end = clock_t::now();
    std::vector<call_t> calls(samples.size());
    const auto stage_start = clock_t::now();
    const auto bv = temporal_enabled ? coordinator_.baseline_views() : cost_baseline_.mutable_views();
    const auto tv = backend_.mutable_views();
    for (auto &c : calls) {
      c.start = query(D3D11_QUERY_TIMESTAMP);
      c.end = query(D3D11_QUERY_TIMESTAMP);
      c.captures["uploaded_source.rgba16f"] = prepare_capture(source_texture_.value.Get());
      c.captures["uploaded_probability.f32"] = prepare_capture(p_buffer_.value.Get());
      c.captures["uploaded_current_support.u32"] = prepare_capture(support_texture_.value.Get());
      c.captures["uploaded_ocr_record.u32"] = prepare_capture(ocr_buffer_.value.Get());
      c.captures["uploaded_constants.u32"] = prepare_capture(constants_buffer_.Get());
      c.captures["uploaded_root_arguments.u32"] = prepare_capture(root_arguments_.Get());
      c.captures["body.u32"] = prepare_capture(temporal_enabled ? tv.body.Get() : bv.body.Get());
      c.captures["final.u32"] = prepare_capture(temporal_enabled ? tv.matte.Get() : bv.matte.Get());
      c.captures["numeric-validation.u32"] = prepare_capture(bv.validation_receipt.Get());
      if (temporal_enabled) {
        c.captures["frame256.u32"] = prepare_capture(tv.frame_constants.Get());
        c.captures["temporal-receipt.u32"] = prepare_capture(tv.receipt.Get());
        c.captures["temporal-state.u32"] = prepare_capture(tv.state.Get());
      }
    }
    auto disjoint = query(D3D11_QUERY_TIMESTAMP_DISJOINT);
    auto prime_start = query(D3D11_QUERY_TIMESTAMP), prime_end = query(D3D11_QUERY_TIMESTAMP);
    capture_t prime_state, prime_receipt;
    if (temporal_enabled) {
      prime_state = prepare_capture(tv.state.Get());
      prime_receipt = prepare_capture(tv.receipt.Get());
    }
    const auto stage_end = clock_t::now();
    upload(samples.front());
    const std::array<UINT, 3u> prime_root {0u, 1u, 1u};
    context_->UpdateSubresource(root_arguments_.Get(), 0u, nullptr, prime_root.data(), 0u, 0u);
    context_->Begin(disjoint.Get());
    context_->End(prime_start.Get());
    const auto prime_cpu_start = clock_t::now();
    need(dispatch_cost(temporal_enabled, samples.front()), error_);
    const auto prime_cpu_end = clock_t::now();
    context_->End(prime_end.Get());
    if (temporal_enabled) {
      copy_capture(prime_state);
      copy_capture(prime_receipt);
    }
    const auto prime_counts = counters(temporal_enabled);
    need(prime_counts.first == 1u && prime_counts.second == 1u, "Root0 primes one cached program only");
    for (std::size_t i = 0u; i < samples.size(); ++i) {
      const auto &s = samples[i];
      auto &c = calls[i];
      // Literal source uploads precede the timestamp. No allocation, staging Map, hash or file IO here.
      upload(s);
      if (measured && s.measured) {
        context_->End(c.start.Get());
      }
      const auto begin = clock_t::now();
      const bool ok = dispatch_cost(temporal_enabled, s);
      const auto end = clock_t::now();
      if (measured && s.measured) {
        context_->End(c.end.Get());
      }
      c.cpu_ns = elapsed(begin, end);
      need(ok, error_);
      const auto count = counters(temporal_enabled);
      c.recordings = count.first;
      c.replays = count.second;
      need(c.recordings == 1u && c.replays == i + 2u, "Steady call rerecorded or skipped its complete recorded function");
      // These diagnostic GPU copies are AFTER the sample-end timestamp and BEFORE next observation.
      for (const auto &[name, capture] : c.captures) {
        copy_capture(capture);
      }
    }
    context_->End(disjoint.Get());
    context_->Flush();
    // All waiting, Maps, hashing and serialization follow the complete sequence interval.
    D3D11_QUERY_DATA_TIMESTAMP_DISJOINT freq {};
    wait_query(disjoint.Get(), &freq, sizeof(freq));
    UINT64 p0 {}, p1 {};
    wait_query(prime_start.Get(), &p0, sizeof(p0));
    wait_query(prime_end.Get(), &p1, sizeof(p1));
    need(!freq.Disjoint && freq.Frequency > 0u && p1 >= p0, "Cost GPU clock invalid; raw evidence retained as failure");
    if (temporal_enabled) {
      const auto st = colour_words(read_archive(prime_state.stage));
      const auto rc = colour_words(read_archive(prime_receipt.stage));
      need(std::all_of(st.begin(), st.end(), [](UINT x) {
             return x == 0u;
           }) &&
             std::all_of(rc.begin(), rc.end(), [](UINT x) {
               return x == 0u;
             }),
           "ROOT0 prime advanced temporal history");
    }
    nlohmann::json rows = nlohmann::json::array();
    for (std::size_t i = 0u; i < samples.size(); ++i) {
      const auto &s = samples[i];
      auto &c = calls[i];
      std::map<std::string, std::vector<char>> data;
      for (const auto &[name, capture] : c.captures) {
        data[name] = read_archive(capture.stage);
      }
      // Preserve literal failed evidence before any functional equality assertion can throw.
      nlohmann::json artifacts;
      if (measured) {
        for (const auto &[name, bytes] : data) {
          artifacts[name] = save(destination / ("case_" + std::to_string(i + 1u)) / name, bytes);
        }
        std::ofstream identity(destination / ("case_" + std::to_string(i + 1u)) / "case-input.json");
        identity << s.item.dump(2) << '\n';
        need(bool(identity), "Cost case identity write failed");
      } else {
        for (const auto &[name, bytes] : data) {
          artifacts[name] = {{"sha256", colour_sha(bytes.data(), bytes.size())}, {"bytes", bytes.size()}};
          if (name == "numeric-validation.u32" || name == "frame256.u32" || name == "temporal-receipt.u32" || name == "temporal-state.u32") {
            artifacts[name]["words_uint32"] = colour_words(bytes);
          }
        }
      }
      need(data.at("uploaded_source.rgba16f") == s.rgb && data.at("uploaded_probability.f32") == s.p && data.at("uploaded_current_support.u32") == s.s && data.at("uploaded_ocr_record.u32") == s.ocr && data.at("uploaded_constants.u32") == s.cb && data.at("uploaded_root_arguments.u32") == s.root, "Cost literal GPU input upload mismatch");
      need(data.at("body.u32") == (temporal_enabled ? s.temporal_body : s.base_body) && data.at("final.u32") == (temporal_enabled ? s.temporal_matte : s.base_matte), "Cost mask changed; no expected-mask waiver");
      const auto v = colour_words(data.at("numeric-validation.u32"));
      const auto cb_words = colour_words(s.cb);
      need(v.size() == 32u && v[0] == 0x31564c49u && v[1] == 1u && v[2] == 1u && v[3] == 0u && std::equal(cb_words.begin(), cb_words.end(), v.begin() + 4u) && std::all_of(v.begin() + 28u, v.end(), [](UINT x) {
             return x == 0u;
           }),
           "Fresh whole-input numeric validation missing");
      if (temporal_enabled) {
        const auto f = colour_words(s.frame);
        const auto r = colour_words(data.at("temporal-receipt.u32"));
        need(data.at("frame256.u32") == s.frame && r.size() == 256u && r[0] == 0x31545742u && r[1] == 1u && r[2] == 1u && r[3] == 0u && r[8] <= 1u && r[9] == 0u && r[10] == s.components, "Temporal cost used cache/root0/unavailable/incomplete work");
        need(std::equal(f.begin() + 4u, f.begin() + 8u, r.begin() + 4u) && std::equal(cb_words.begin(), cb_words.end(), r.begin() + 52u) && std::equal(f.begin() + 48u, f.begin() + 56u, r.begin() + 76u), "Source/model/continuity changed during cost");
      }
      nlohmann::json ticks = nullptr;
      if (measured && s.measured) {
        UINT64 begin {}, end {};
        wait_query(c.start.Get(), &begin, sizeof(begin));
        wait_query(c.end.Get(), &end, sizeof(end));
        need(end >= begin, "GPU timestamp regressed");
        ticks = {begin, end};
      }
      rows.push_back({{"case", s.item}, {"measured_positive", measured && s.measured}, {"source_positive", s.measured}, {"ticks_uint64", ticks}, {"cpu_full_function_submit_ns", c.cpu_ns}, {"recordings", c.recordings}, {"replays", c.replays}, {"artifacts", artifacts}});
    }
    return {{"name", sequence.at("name")}, {"fresh_physical_backend_scope", true}, {"warm_play", warm_play}, {"measured_replay", measured}, {"allocation_CPU_ns_O0_diagnostic", elapsed(allocation_start, allocation_end)}, {"staging_query_allocation_CPU_ns", elapsed(stage_start, stage_end)}, {"prime_CPU_ns_O0_diagnostic", elapsed(prime_cpu_start, prime_cpu_end)}, {"prime_root", prime_root}, {"prime_ticks_uint64", {p0, p1}}, {"prime_temporal_history_all_zero", temporal_enabled}, {"frequency_hz", freq.Frequency}, {"disjoint", freq.Disjoint != FALSE}, {"calls", rows}};
  }

  base_t cost_baseline_;
};

TEST_F(HostSbsBodyWitnessTemporalCostGpuTest, CompleteOriginal28RecordedCost) {
  const char *out = std::getenv("SUNSHINE_TEST_BODY_WITNESS_TEMPORAL_COST_OUTPUT");
  const char *plan = std::getenv("SUNSHINE_TEST_BODY_WITNESS_TEMPORAL_COST_PLAN");
  if (!out || !plan || spec_path_.empty()) {
    GTEST_SKIP() << "Explicit frozen original28 cost plan/cases/output required";
  }
  const auto plan_bytes = colour_bytes(plan);
  const auto cost_plan = nlohmann::json::parse(plan_bytes.begin(), plan_bytes.end());
  const std::filesystem::path output(out);
  ASSERT_FALSE(std::filesystem::exists(output));
  ASSERT_TRUE(std::filesystem::create_directories(output));
  std::array<wchar_t, 32768u> executable_name {};
  const auto executable_length = GetModuleFileNameW(nullptr, executable_name.data(), DWORD(executable_name.size()));
  ASSERT_GT(executable_length, 0u);
  ASSERT_LT(executable_length, executable_name.size());
  const std::filesystem::path executable_path(std::wstring(executable_name.data(), executable_length));
  D3D11_FEATURE_DATA_THREADING threading {};
  const HRESULT threading_hr = device_->CheckFeatureSupport(D3D11_FEATURE_THREADING, &threading, sizeof(threading));
  nlohmann::json report {{"schema", 1}, {"status", "INCOMPLETE"}, {"input", save(output / "literal-input.json", colour_bytes(spec_path_))}, {"plan", save(output / "literal-cost-plan.json", plan_bytes)}, {"actual_executable_path", executable_path.string()}, {"driver", driver_}, {"adapter", adapter_identity()}, {"driver_threading_query_succeeded", SUCCEEDED(threading_hr)}, {"driver_native_command_lists", SUCCEEDED(threading_hr) && threading.DriverCommandLists != FALSE}, {"strategies", nlohmann::json::array()}, {"all_semantic_status_UNKNOWN", true}, {"CPU_test_O0_not_live_latency", true}};
  try {
    need(input_spec_.at("driver").get<std::string>() == driver_ && cost_plan.at("warm_full_replays").get<UINT>() == 3u && cost_plan.at("measured_positive_calls").get<UINT>() == 20u && cost_plan.at("untimed_empty_calls").get<UINT>() == 8u, "Cost protocol differs frozen3/20/8");
    need(temporal_closure_ == "1d6d0c31605389a6328fd9636139db8dd7414bb08df22c833d2eab46eb9b9a74", "Cost requires parity-proven immutable-palette metadata source");
    std::vector<std::vector<sample_t>> samples;
    UINT positives = 0u, total = 0u;
    for (const auto &sequence : input_spec_.at("sequences")) {
      samples.push_back(load_samples(sequence));
      for (const auto &s : samples.back()) {
        positives += s.measured ? 1u : 0u;
        ++total;
      }
    }
    need(total == 28u && positives == 20u, "Original28 workload must contain20positive/8empty");
    for (bool temporal_enabled : {false, true}) {
      const std::string strategy = temporal_enabled ? "temporal_owned_recorded" : "baseline5cad_recorded";
      nlohmann::json r {{"name", strategy}, {"include_baseline_control", false}, {"warm_replays", nlohmann::json::array()}, {"measured_sequences", nlohmann::json::array()}};
      for (UINT play = 0u; play < 3u; ++play) {
        nlohmann::json warm = nlohmann::json::array();
        for (std::size_t k = 0u; k < samples.size(); ++k) {
          warm.push_back(run_scope(temporal_enabled, input_spec_.at("sequences")[k], samples[k], {}, false, play));
        }
        r["warm_replays"].push_back(std::move(warm));
      }
      for (std::size_t k = 0u; k < samples.size(); ++k) {
        const auto &sequence = input_spec_.at("sequences")[k];
        r["measured_sequences"].push_back(run_scope(temporal_enabled, sequence, samples[k], output / strategy / sequence.at("name").get<std::string>(), true, 0u));
      }
      report["strategies"].push_back(std::move(r));
    }
    report["status"] = "PASS_FUNCTIONAL_COST_RAW_NOT_ADOPTION";
  } catch (const std::exception &e) {
    report["status"] = "FAIL";
    report["failure"] = e.what();
    ADD_FAILURE() << e.what();
  }
  std::ofstream file(output / "raw-cost-report.json");
  file << report.dump(2) << '\n';
  ASSERT_TRUE(file.good());
}
