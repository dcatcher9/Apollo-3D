// Additive whole-function cost fixture. Original mechanics/Saved28 and header remain literal.
// All allocation, ROOT0 cache prime, warmup, query reads, copies/Maps and hashing are outside
// measured intervals. No model, current OCR generation, renderer, live or end-to-end cost.

#include <chrono>
#include <thread>

namespace {

  class HostSbsCurrentBlackEnclosureCostGpuTest: public HostSbsCurrentBlackEnclosureGpuTest {
  protected:
    using clock_t = std::chrono::steady_clock;

    struct sample_t {
      UINT strategy = 0u, block = 0u, ordinal = 0u;
      std::uint64_t submit_ns = 0u, start_tick = 0u, end_tick = 0u;
    };

    void SetUp() override {
      const auto *flag = std::getenv("SUNSHINE_TEST_BLACK_ENCLOSURE_COST");
      if (!flag || std::string_view(flag) != "1" || !std::getenv("SUNSHINE_TEST_BLACK_ENCLOSURE_CASES") || !std::getenv("SUNSHINE_TEST_BLACK_ENCLOSURE_COST_OUTPUT")) {
        GTEST_SKIP() << "Explicit whole-function cost opt-in/cases/output required";
      }
      HostSbsCurrentBlackEnclosureGpuTest::SetUp();
    }

    bool submit(UINT strategy) {
      if (strategy == 0u) {
        return literal_.dispatch(context_.Get(), colour_ptrs(), graph_ptrs(), cb_.resource.Get(), rgb_.srv.Get(), probability_.srv.Get(), support_.srv.Get(), ocr_.srv.Get(), root_.resource.Get(), 240u, error_);
      }
      return run(true, true);
    }

    nlohmann::json counters(UINT strategy) const {
      if (strategy == 0u) {
        const auto d = literal_.execution_diagnostics();
        return {{"recordings", d.recordings}, {"replays", d.replays}, {"direct_emissions", 0u}, {"predication_direct_fallbacks", d.predication_direct_fallbacks}, {"allocation_replacements", d.allocation_replacements}};
      }
      const auto d = helper_.diagnostics();
      return {{"recordings", d.recordings}, {"replays", d.replays}, {"direct_emissions", d.direct_emissions}, {"predication_direct_fallbacks", 0u}, {"allocation_replacements", d.allocation_replacements}, {"unavailable", d.unavailable}};
    }

    void fresh_helpers() {
      helper_ = enclosure::resources_t {};
      literal_ = baseline::resources_t {};
      literal_.set_recorded_execution(true);
    }

    void prepare_literal() {
      require(literal_.ensure(device_.Get(), W_, H_, error_));
    }

    void load_saved(const nlohmann::json &item) {
      fresh_helpers();
      const auto &a = item.at("artifacts");
      const auto cb = bound_bytes(a.at("constants"), 96u);
      std::memcpy(cb_words_.data(), cb.data(), 96u);
      const auto &producer = item.at("producer");
      const UINT ow = producer.at("origin_width"), oh = producer.at("origin_height");
      allocate_inputs(cb_words_[0], cb_words_[3], cb_words_[1]);
      const std::size_t n = std::size_t(W_) * H_;
      const auto source = bound_bytes(a.at("source_band"), n * 8u), p = bound_bytes(a.at("probability"), 614400u), s = bound_bytes(a.at("current_support"), n * 4u), o = bound_bytes(a.at("ocr_record"), 1024u), root = bound_bytes(a.at("root_arguments"), 256u), sct = bound_bytes(a.at("current_sct"), 128u), uip = bound_bytes(a.at("current_uip"), 160u), origin = bound_bytes(a.at("original_source_rgba8"), std::size_t(ow) * oh * 4u);
      require(producer.at("origin_format").get<UINT>() == 28u);
      origin_ = texture(ow, oh, DXGI_FORMAT_R8G8B8A8_UNORM);
      context_->UpdateSubresource(rgb_.resource.Get(), 0u, nullptr, source.data(), W_ * 8u, 0u);
      context_->UpdateSubresource(support_.resource.Get(), 0u, nullptr, s.data(), W_ * 4u, 0u);
      context_->UpdateSubresource(probability_.resource.Get(), 0u, nullptr, p.data(), 0u, 0u);
      context_->UpdateSubresource(ocr_.resource.Get(), 0u, nullptr, o.data(), 0u, 0u);
      context_->UpdateSubresource(cb_.resource.Get(), 0u, nullptr, cb.data(), 0u, 0u);
      context_->UpdateSubresource(root_.resource.Get(), 0u, nullptr, root.data(), 0u, 0u);
      context_->UpdateSubresource(sct_.resource.Get(), 0u, nullptr, sct.data(), 0u, 0u);
      context_->UpdateSubresource(uip_.resource.Get(), 0u, nullptr, uip.data(), 0u, 0u);
      context_->UpdateSubresource(origin_.resource.Get(), 0u, nullptr, origin.data(), ow * 4u, 0u);
      program_.current_sct = sct_.srv.Get();
      program_.current_uip = uip_.srv.Get();
      auto &pr = program_.producer;
      pr.original_source = origin_.srv.Get();
      pr.transfer = static_cast<enclosure::transfer_t>(producer.at("transfer").get<UINT>());
      pr.input_known = producer.at("input_known");
      pr.direct_encoded_recipe = producer.at("direct_encoded_recipe");
      pr.postprocess_mode = producer.at("postprocess_mode");
      pr.expected_work = producer.at("expected_work");
      pr.transaction_token = producer.at("transaction_token");
      pr.model_frame = producer.at("model_frame");
      pr.model_observation_us = producer.at("model_observation_us");
      pr.analysis_generation = producer.at("analysis_generation");
      pr.analysis_roi = producer.at("analysis_roi").get<std::array<UINT, 4u>>();
      literal_root_ = root;
      for (const auto &[file, key] : std::array<std::pair<const char *, const char *>, 4u> {{{"baseline_body.u32", "baseline_body"}, {"baseline_final.u32", "baseline_final"}, {"body.u32", "body"}, {"final.u32", "final"}}}) {
        expected_[file] = bound_bytes(item.at("expected").at(key), n * 4u);
      }
      require(item.at("expected").contains("receipt"));
      expected_["receipt.u32"] = bound_bytes(item.at("expected").at("receipt"), 2048u);
      prepare_literal();
    }

    void load_stress(UINT stress) {
      fresh_helpers();
      author(true, 9000u + stress, 1u, false, false, false, 256u, 128u, stress);
      literal_root_ = raw(root_words_);
      expected_.clear();
      prepare_literal();
    }

    ComPtr<ID3D11Query> query(D3D11_QUERY type) {
      D3D11_QUERY_DESC d {type, 0u};
      ComPtr<ID3D11Query> q;
      require(SUCCEEDED(device_->CreateQuery(&d, &q)));
      return q;
    }

    template<class T>
    void wait_query(ID3D11Query *q, T &value) {
      const auto deadline = clock_t::now() + std::chrono::seconds(60);
      while (true) {
        const HRESULT hr = context_->GetData(q, &value, sizeof(value), 0u);
        if (hr == S_OK) {
          return;
        }
        require(hr == S_FALSE && clock_t::now() < deadline);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }

    nlohmann::json save_arrays(const std::filesystem::path &folder, const std::map<std::string, std::vector<char>> &arrays) {
      nlohmann::json result = nlohmann::json::object();
      for (const auto &[name, data] : arrays) {
        std::ofstream f(folder / name, std::ios::binary);
        f.write(data.data(), static_cast<std::streamsize>(data.size()));
        require(f.good());
        result[name] = {{"path", (folder / name).string()}, {"sha256", hash_bytes(data)}, {"bytes", data.size()}};
      }
      return result;
    }

    nlohmann::json execute_case(const std::string &name, UINT ordinal, bool authored, const nlohmann::json &input_case, const std::filesystem::path &output) {
      const auto case_start = clock_t::now();
      const auto allocation_done = clock_t::now();  // Caller just completed fresh allocation/upload.
      std::array<ComPtr<ID3D11Query>, 40u> begins, ends;
      auto disjoint = query(D3D11_QUERY_TIMESTAMP_DISJOINT);
      for (UINT i = 0u; i < 40u; ++i) {
        begins[i] = query(D3D11_QUERY_TIMESTAMP);
        ends[i] = query(D3D11_QUERY_TIMESTAMP);
      }
      const auto prime_begin = clock_t::now();
      auto held = words(literal_root_);
      require(held.size() == 64u && held[60] == 1u && held[61] == 1u && held[62] == 1u);
      held[60] = 0u;
      context_->UpdateSubresource(root_.resource.Get(), 0u, nullptr, held.data(), 0u, 0u);
      require(submit(0u));
      require(submit(1u));
      const auto private_v = helper_.mutable_views();
      require(read(private_v.receipt.Get()) == std::vector<char>(2048u, 0));
      require(body() == std::vector<char>(std::size_t(W_) * H_ * 4u, 0));
      require(matte() == std::vector<char>(std::size_t(W_) * H_ * 4u, 0));
      context_->UpdateSubresource(root_.resource.Get(), 0u, nullptr, literal_root_.data(), 0u, 0u);
      const auto prime_end = clock_t::now();
      const auto warm_begin = clock_t::now();
      for (UINT i = 0u; i < 3u; ++i) {
        require(submit(ordinal & 1u));
        require(submit(1u - (ordinal & 1u)));
      }
      std::map<std::string, std::vector<char>> warm {{"baseline_body.u32", read(literal_.mutable_views().body.Get())}, {"baseline_final.u32", read(literal_.mutable_views().matte.Get())}, {"body.u32", body()}, {"final.u32", matte()}, {"receipt.u32", read(private_v.receipt.Get())}};
      const auto warm_end = clock_t::now();
      if (authored) {
        expected_ = warm;  // Stability only; numeric mechanics are separately qualified.
      }
      const auto before0 = counters(0u), before1 = counters(1u);
      require(before0.at("recordings") == 1u && before1.at("recordings") == 1u && before0.at("replays") == 4u && before1.at("replays") == 4u);
      std::array<sample_t, 40u> samples {};
      const std::array<UINT, 4u> order = ordinal & 1u ? std::array<UINT, 4u> {1u, 0u, 0u, 1u} : std::array<UINT, 4u> {0u, 1u, 1u, 0u};
      const auto measured_begin = clock_t::now();
      context_->Begin(disjoint.Get());
      for (UINT block = 0u; block < 4u; ++block) {
        for (UINT repeat = 0u; repeat < 10u; ++repeat) {
          const UINT i = block * 10u + repeat;
          auto &sample = samples[i];
          sample.strategy = order[block];
          sample.block = block;
          sample.ordinal = repeat;
          context_->End(begins[i].Get());
          const auto cpu_start = clock_t::now();
          const bool ok = submit(sample.strategy);
          const auto cpu_end = clock_t::now();
          sample.submit_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_end - cpu_start).count());
          context_->End(ends[i].Get());
          require(ok);
        }
      }
      context_->End(disjoint.Get());
      const auto measured_submitted = clock_t::now();
      context_->Flush();  // Only after every end timestamp/disjoint was queued.
      D3D11_QUERY_DATA_TIMESTAMP_DISJOINT timing {};
      wait_query(disjoint.Get(), timing);
      for (UINT i = 0u; i < 40u; ++i) {
        wait_query(begins[i].Get(), samples[i].start_tick);
        wait_query(ends[i].Get(), samples[i].end_tick);
      }
      const auto queries_read = clock_t::now();
      const auto after0 = counters(0u), after1 = counters(1u);
      const bool cached = after0.at("recordings") == before0.at("recordings") && after1.at("recordings") == before1.at("recordings") && after0.at("replays").get<std::uint64_t>() - before0.at("replays").get<std::uint64_t>() == 20u && after1.at("replays").get<std::uint64_t>() - before1.at("replays").get<std::uint64_t>() == 20u && after1.at("direct_emissions") == 0u && after0.at("predication_direct_fallbacks") == 0u;
      std::map<std::string, std::vector<char>> arrays {{"uploaded_source.rgba16f", read(rgb_.resource.Get())}, {"uploaded_probability.f32", read(probability_.resource.Get())}, {"uploaded_current_support.u32", read(support_.resource.Get())}, {"uploaded_ocr_record.u32", read(ocr_.resource.Get())}, {"uploaded_constants.u32", read(cb_.resource.Get())}, {"uploaded_root.u32", read(root_.resource.Get())}, {"uploaded_SCT.u32", read(sct_.resource.Get())}, {"uploaded_UIP.u32", read(uip_.resource.Get())}, {"uploaded_origin.rgba8", read(origin_.resource.Get())}, {"producer_context.u32", read(private_v.producer_constants.Get())}, {"root_mirror.u32", read(private_v.root_mirror.Get())}, {"baseline_body.u32", read(literal_.mutable_views().body.Get())}, {"baseline_final.u32", read(literal_.mutable_views().matte.Get())}, {"enabled_baseline_body.u32", read(helper_.baseline_views().body.Get())}, {"enabled_baseline_final.u32", read(helper_.baseline_views().matte.Get())}, {"body.u32", body()}, {"final.u32", matte()}, {"receipt.u32", read(private_v.receipt.Get())}};
      bool exact = arrays.at("baseline_body.u32") == arrays.at("enabled_baseline_body.u32") && arrays.at("baseline_final.u32") == arrays.at("enabled_baseline_final.u32") && arrays.at("uploaded_root.u32") == literal_root_;
      for (const auto &[key, expected] : expected_) {
        exact = exact && arrays.at(key) == expected && warm.at(key) == expected;
      }
      for (const auto &[key, value] : warm) {
        arrays["warm_" + key] = value;
      }
      const auto folder = output / ("case_" + std::to_string(ordinal + 1u));
      require(std::filesystem::create_directory(folder));
      const auto artifacts = save_arrays(folder, arrays);
      nlohmann::json raw_samples = nlohmann::json::array();
      for (const auto &s : samples) {
        raw_samples.push_back({{"strategy", s.strategy}, {"block", s.block}, {"ordinal", s.ordinal}, {"start_tick", s.start_tick}, {"end_tick", s.end_tick}, {"CPU_submit_ns", s.submit_ns}});
      }
      const auto ns = [&](clock_t::time_point t) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(t - case_start).count();
      };
      return {{"schema", 1u}, {"name", name}, {"ordinal", ordinal}, {"authored_stress", authored}, {"case", input_case}, {"shape", {W_, H_}}, {"source_model_identity_unchanged", true}, {"warmups_per_strategy", 3u}, {"ROOT0_primes_per_strategy", 1u}, {"measured_per_strategy", 20u}, {"blocks", order}, {"block_repeats", 10u}, {"frequency", timing.Frequency}, {"disjoint", timing.Disjoint != FALSE}, {"samples", raw_samples}, {"counters_before", {before0, before1}}, {"counters_after", {after0, after1}}, {"all_measured_cached_recorded", cached}, {"functional_end_exact", exact}, {"functional_reference", authored ? "AUTHORED_MECHANICS_PLUS_WARM_STABILITY_ONLY" : "FROZEN_SAVED28_CPU_CANDIDATE_AND_NATIVE_BASELINE_AND_FULL_META"}, {"artifacts", artifacts}, {"phase_ns_from_case_start", {{"allocation_already_done", ns(allocation_done)}, {"prime_begin", ns(prime_begin)}, {"prime_end", ns(prime_end)}, {"warm_begin", ns(warm_begin)}, {"warm_end", ns(warm_end)}, {"measured_begin", ns(measured_begin)}, {"measured_submission_end", ns(measured_submitted)}, {"queries_read", ns(queries_read)}}}, {"no_diagnostic_Map_CopyResource_query_read_or_hash_in_timestamp_intervals", true}, {"baseline_dispatch_command_calls", 458u}, {"enabled_dispatch_command_calls", 599u}, {"new_persistent_payload_bytes", std::uint64_t(20u) * W_ * H_ + 2572u}, {"qualified_performance_comparison", false}, {"CPU_submit_scope", "NATIVE_TEST_CXXO2_WITH_COVERAGE_VERIFIED_BY_EXTERNAL_COMPILE_PROVENANCE_NOT_LIVE"}};
    }

    void cost() {
      const std::filesystem::path input = std::getenv("SUNSHINE_TEST_BLACK_ENCLOSURE_CASES"), output = std::getenv("SUNSHINE_TEST_BLACK_ENCLOSURE_COST_OUTPUT");
      const auto input_bytes = file_bytes(input);
      const auto specification = nlohmann::json::parse(input_bytes.begin(), input_bytes.end());
      ASSERT_EQ(specification.at("schema"), 1u);
      ASSERT_EQ(specification.at("driver"), hardware_ ? "hardware" : "warp");
      ASSERT_EQ(specification.at("cases").size(), 28u);
      ASSERT_EQ(specification.at("inner_source_closure_sha256"), inner_closure_);
      ASSERT_EQ(specification.at("component_source_closure_sha256"), graph_closure_);
      ASSERT_EQ(specification.at("enclosure_source_closure_sha256"), enclosure_closure_);
      ASSERT_EQ(enclosure_closure_, "5a0c65be47d7adaf496a1a9d26b2ba5c81e2491341b0fee4e515ae89f9bb8a35");
      ASSERT_FALSE(std::filesystem::exists(output));
      ASSERT_TRUE(std::filesystem::create_directories(output));
      std::array<wchar_t, 32768u> path {};
      const auto len = GetModuleFileNameW(nullptr, path.data(), DWORD(path.size()));
      ASSERT_GT(len, 0u);
      ASSERT_LT(len, path.size());
      const std::filesystem::path exe(std::wstring(path.data(), len));
      const auto exe_bytes = file_bytes(exe);
      nlohmann::json report {{"schema", 1u}, {"scope", "WHOLE_RECORDED_HELPER_ONLY_PRIVATE5CAD_COMPARISON_NOT_PUBLISHED121_MODEL_RENDER_OR_LIVE"}, {"driver", hardware_ ? "hardware" : "warp"}, {"adapter", adapter_}, {"test_executable", {{"path", exe.string()}, {"sha256", hash_bytes(exe_bytes)}, {"bytes", exe_bytes.size()}}}, {"input_specification", {{"path", input.string()}, {"sha256", hash_bytes(input_bytes)}, {"bytes", input_bytes.size()}}}, {"helper_source_pin", compiled_enclosure_helper_pin}, {"baseline_helper_source_pin", compiled_baseline_helper_pin}, {"enclosure_source_closure_sha256", enclosure_closure_}, {"inner_source_closure_sha256", inner_closure_}, {"component_source_closure_sha256", graph_closure_}, {"actual_case_count", 28u}, {"authored_stress_count", 2u}, {"new_algorithm_or_expected_masks", false}, {"all_semantic_status_UNKNOWN", true}, {"qualified_performance_comparison", false}, {"cases", nlohmann::json::array()}};
      const auto write_report = [&] {
        std::ofstream f(output / "current_black_enclosure_cost.json", std::ios::binary | std::ios::trunc);
        f << report.dump(2) << '\n';
        require(f.good());
      };
      for (UINT i = 0u; i < 30u; ++i) {
        const auto load_start = clock_t::now();
        const bool authored = i >= 28u;
        nlohmann::json item;
        if (!authored) {
          item = specification.at("cases").at(i);
          expected_.clear();
          load_saved(item);
        } else {
          load_stress(i - 27u);
          item = {{"name", i == 28u ? "AUTHORED_dense_256x128" : "AUTHORED_snake_256x128"}, {"scope", "UNCHANGED_NATIVE_MECHANICS_SOURCE_STRESS_NOT_ACTUAL28_OR_GLYPH_QUALITY"}};
        }
        const auto prepared = clock_t::now();
        auto row = execute_case(item.at("name").get<std::string>(), i, authored, item, output);
        row["preallocation_and_upload_ns"] = std::chrono::duration_cast<std::chrono::nanoseconds>(prepared - load_start).count();
        EXPECT_TRUE(row.at("all_measured_cached_recorded").get<bool>());
        EXPECT_TRUE(row.at("functional_end_exact").get<bool>());
        EXPECT_FALSE(row.at("disjoint").get<bool>());
        EXPECT_GT(row.at("frequency").get<std::uint64_t>(), 0u);
        report["cases"].push_back(std::move(row));
        write_report();
      }
    }

    baseline::resources_t literal_;
    std::vector<char> literal_root_;
    std::map<std::string, std::vector<char>> expected_;
  };

  TEST_F(HostSbsCurrentBlackEnclosureCostGpuTest, CompleteOriginal28AndDenseSnakeWholeFunctionRecordedCost) {
    cost();
  }
}  // namespace
