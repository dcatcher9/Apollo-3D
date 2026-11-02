// Additive whole-function cost fixture. Original mechanics/Saved28 and header remain literal.
// All allocation, ROOT0 cache prime, warmup, query reads, copies/Maps and hashing are outside
// measured intervals. No model, current OCR generation, renderer, live or end-to-end cost.

#include <chrono>
#include <thread>

namespace {

  class HostSbsCurrentOutlineSupportCostGpuTest: public HostSbsCurrentOutlineSupportClosureGpuTest {
  protected:
    using clock_t = std::chrono::steady_clock;

    struct sample_t {
      UINT strategy = 0u, block = 0u, ordinal = 0u;
      std::uint64_t submit_ns = 0u, start_tick = 0u, end_tick = 0u;
      bool submitted = false, start_ready = false, end_ready = false;
      HRESULT start_status = S_FALSE, end_status = S_FALSE;
      std::string error;
    };

    void SetUp() override {
      const auto *flag = std::getenv("SUNSHINE_TEST_OUTLINE_CLOSURE_COST");
      if (!flag || std::string_view(flag) != "1" || !std::getenv("SUNSHINE_TEST_OUTLINE_CLOSURE_CASES") || !std::getenv("SUNSHINE_TEST_OUTLINE_CLOSURE_COST_OUTPUT")) {
        GTEST_SKIP() << "Explicit whole-function cost opt-in/cases/output required";
      }
      HostSbsCurrentOutlineSupportClosureGpuTest::SetUp();
    }

    bool submit(UINT strategy) {
      if (strategy == 0u) {
        return literal_.dispatch(context_.Get(), colour_ptrs(), graph_ptrs(), cb_.resource.Get(), rgb_.srv.Get(), probability_.srv.Get(), support_.srv.Get(), ocr_.srv.Get(), root_.resource.Get(), 240u, error_);
      }
      return helper_.dispatch(context_.Get(), colour_ptrs(), graph_ptrs(), cb_.resource.Get(), rgb_.srv.Get(), probability_.srv.Get(), support_.srv.Get(), ocr_.srv.Get(), root_.resource.Get(), 240u, error_, &program_);
    }

    nlohmann::json counters() const {
      const auto l = literal_.execution_diagnostics();
      const auto p = helper_.parent_diagnostics(), a = helper_.diagnostics();
      const auto own = [](const auto &d) {
        return nlohmann::json {{"recordings", d.recordings}, {"replays", d.replays}, {"direct_emissions", d.direct_emissions}, {"allocation_replacements", d.allocation_replacements}, {"unavailable", d.unavailable}};
      };
      return {{"literal", {{"recordings", l.recordings}, {"replays", l.replays}, {"predication_direct_fallbacks", l.predication_direct_fallbacks}, {"allocation_replacements", l.allocation_replacements}}}, {"CBC", own(p)}, {"appendix", own(a)}};
    }

    void fresh_helpers() {
      helper_ = ring::resources_t {};
      helper_.set_recorded_execution(true);
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
      program_.enclosure.current_sct = sct_.srv.Get();
      program_.enclosure.current_uip = uip_.srv.Get();
      auto &pr = program_.enclosure.producer;
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
      program_.packed_temp_snapshot = nullptr;
      program_.passable_cache_snapshot = nullptr;
      packed_snapshot_ = {};
      cache_snapshot_ = {};
      prepare_literal();
    }

    void load_stress(UINT stress) {
      fresh_helpers();
      author(true, 9000u + stress, 1u, false, false, false, 256u, 128u, stress);
      literal_root_ = raw(root_words_);
      expected_.clear();
      program_.packed_temp_snapshot = nullptr;
      program_.passable_cache_snapshot = nullptr;
      packed_snapshot_ = {};
      cache_snapshot_ = {};
      prepare_literal();
    }

    ComPtr<ID3D11Query> query(D3D11_QUERY type) {
      D3D11_QUERY_DESC d {type, 0u};
      ComPtr<ID3D11Query> q;
      require(SUCCEEDED(device_->CreateQuery(&d, &q)));
      return q;
    }

    template<class T>
    bool get_query(ID3D11Query *q, T &v, clock_t::time_point deadline, HRESULT &status) {
      do {
        status = context_->GetData(q, &v, sizeof(v), 0u);
        if (status == S_OK) {
          return true;
        }
        if (status != S_FALSE) {
          return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      } while (clock_t::now() < deadline);
      return false;
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

    using arrays_t = std::map<std::string, std::vector<char>>;

    arrays_t controls(ring::resources_t &owner) {
      const auto e = owner.enclosure_views();
      const auto b = owner.baseline_views();
      const auto r = owner.raw_views();
      const auto v = owner.mutable_views();
      return {{"baseline_body.u32", read(b.body.Get())}, {"baseline_final.u32", read(b.matte.Get())}, {"body.u32", read(e.body.Get())}, {"final.u32", read(e.matte.Get())}, {"receipt.u32", read(e.receipt.Get())}, {"all_body_unit_bits.u32", read(e.all_body_bits.Get())}, {"cover_unit_bits.u32", read(e.cover_bits.Get())}, {"selected_unit_bits.u32", read(e.selected_bits.Get())}, {"numeric_validation.u32", read(b.validation_receipt.Get())}, {"unit_arguments.u32", read(e.unit_arguments.Get())}, {"ring_final.u32", read(r.matte.Get())}, {"ring_receipt.u32", read(r.receipt.Get())}, {"ring_eligible_inner.u32", read(r.eligible_inner.Get())}, {"ring_unit_arguments.u32", read(r.unit_arguments.Get())}, {"ring_native_arguments.u32", read(r.native_arguments.Get())}, {"outline_final.u32", read(v.matte.Get())}, {"calibration_data.u32", read(v.calibration_data.Get())}, {"passable_root_cache.u32", read(v.passable_root_cache.Get())}};
    }

    arrays_t capture() {
      auto a = controls(helper_);
      const auto e = helper_.enclosure_views();
      a.emplace("uploaded_source.rgba16f", read(rgb_.resource.Get()));
      a.emplace("uploaded_probability.f32", read(probability_.resource.Get()));
      a.emplace("uploaded_current_support.u32", read(support_.resource.Get()));
      a.emplace("uploaded_ocr_record.u32", read(ocr_.resource.Get()));
      a.emplace("uploaded_constants.u32", read(cb_.resource.Get()));
      a.emplace("uploaded_root.u32", read(root_.resource.Get()));
      a.emplace("uploaded_SCT.u32", read(sct_.resource.Get()));
      a.emplace("uploaded_UIP.u32", read(uip_.resource.Get()));
      a.emplace("uploaded_origin.rgba8", read(origin_.resource.Get()));
      a.emplace("producer_context.u32", read(e.producer_constants.Get()));
      a.emplace("root_mirror.u32", read(e.root_mirror.Get()));
      a.emplace("literal_body.u32", read(literal_.mutable_views().body.Get()));
      a.emplace("literal_final.u32", read(literal_.mutable_views().matte.Get()));
      require(a.size() == 31u);
      return a;
    }

    bool matches(const arrays_t &a, const arrays_t &expected) const {
      for (const auto &[key, value] : expected) {
        const auto found = a.find(key);
        if (found == a.end() || found->second != value) {
          return false;
        }
      }
      return true;
    }

    nlohmann::json hashes(const arrays_t &a) {
      auto j = nlohmann::json::object();
      for (const auto &[key, value] : a) {
        j[key] = {{"sha256", hash_bytes(value)}, {"bytes", value.size()}};
      }
      return j;
    }

    arrays_t direct_control() {
      ring::resources_t reference;
      require(reference.ensure(device_.Get(), W_, H_, error_));
      reference.set_recorded_execution(false);
      require(reference.dispatch(context_.Get(), colour_ptrs(), graph_ptrs(), cb_.resource.Get(), rgb_.srv.Get(), probability_.srv.Get(), support_.srv.Get(), ocr_.srv.Get(), root_.resource.Get(), 240u, error_, &program_));
      auto a = controls(reference);
      const auto d = reference.diagnostics(), p = reference.parent_diagnostics();
      require(d.recordings == 0u && d.direct_emissions == 1u && p.recordings == 0u && p.direct_emissions == 1u);
      return a;
    }

    arrays_t expected_hold(const arrays_t &active) {
      auto a = active;
      auto r = words(literal_root_);
      r[60u] = 0u;
      a["uploaded_root.u32"] = raw(r);
      a["root_mirror.u32"] = raw(r);
      a["numeric_validation.u32"] = std::vector<char>(128u, 0);
      auto args = words(a.at("ring_native_arguments.u32"));
      require(args.size() == 3u);
      args[2u] = 0u;
      a["ring_native_arguments.u32"] = raw(args);
      return a;
    }

    bool stress_parity(const arrays_t &a, UINT stress) {
      const auto body = words(a.at("body.u32")), m = words(a.at("outline_final.u32")), d = words(a.at("calibration_data.u32"));
      bool exact = d.size() == 68608u && d[2u] == 1u && d[3u] == 0u;
      for (UINT y = 0u; y < H_; ++y) {
        for (UINT x = 0u; x < W_; ++x) {
          const bool b = x >= 80u && x < 176u && y >= 48u && y < 88u;
          const bool f = stress == 1u ? (x >= 64u && x < 192u && y >= 32u && y < 104u) : (x >= 77u && x < 179u && y >= 45u && y < 91u);
          exact = exact && body[y * W_ + x] == UINT(b) && m[y * W_ + x] == UINT(f);
        }
      }
      if (stress == 1u) {
        exact = exact && d[139u] == 16u;
      } else {
        exact = exact && d[139u] == 0u && d[150u] == 17u && (d[151u] & 4u) != 0u;
      }
      return exact;
    }

    bool cached_delta(const nlohmann::json &before, const nlohmann::json &after, UINT replay_before) {
      for (const char *owner : {"literal", "CBC", "appendix"}) {
        const auto &b = before.at(owner), &a = after.at(owner);
        if (b.at("recordings") != 1u || a.at("recordings") != 1u || b.at("replays") != replay_before || a.at("replays") != replay_before + 20u || b.at("allocation_replacements") != 0u || a.at("allocation_replacements") != 0u) {
          return false;
        }
        if (std::string_view(owner) == "literal") {
          if (b.at("predication_direct_fallbacks") != 0u || a.at("predication_direct_fallbacks") != 0u) {
            return false;
          }
        } else if (b.at("direct_emissions") != 0u || a.at("direct_emissions") != 0u || b.at("unavailable") != 0u || a.at("unavailable") != 0u) {
          return false;
        }
      }
      return true;
    }

    nlohmann::json phase(const std::string &mode, UINT ordinal, UINT stress, const nlohmann::json &item, const arrays_t &direct, const arrays_t &hold_expected, const std::filesystem::path &folder, arrays_t &final_capture) {
      const bool hold = mode == "root0_hold";
      auto r = words(literal_root_);
      if (hold) {
        r[60u] = 0u;
      }
      context_->UpdateSubresource(root_.resource.Get(), 0u, nullptr, r.data(), 0u, 0u);
      const auto phase_start = clock_t::now();
      for (UINT k = 0u; k < 3u; ++k) {
        require(submit(ordinal & 1u));
        require(submit(1u - (ordinal & 1u)));
      }
      const auto warm = capture();
      const bool parity = hold ? matches(warm, hold_expected) : (matches(warm, expected_) && matches(warm, direct) && (!stress || stress_parity(warm, stress)));
      auto row = nlohmann::json {{"mode", mode}, {"warmups_per_strategy", 3u}, {"measured_per_strategy", 20u}, {"pre_timing_parity_pass", parity}, {"snapshots_null", program_.packed_temp_snapshot == nullptr && program_.passable_cache_snapshot == nullptr}, {"source_is_new_observation", false}, {"samples", nlohmann::json::array()}};
      require(std::filesystem::create_directory(folder));
      if (!parity) {
        row["status"] = "FAIL_PRE_TIMING_PARITY_NO_SAMPLES";
        row["artifacts"] = save_arrays(folder, warm);
        row["warm_hashes"] = hashes(warm);
        final_capture = warm;
        return row;
      }
      const auto before = counters();
      std::array<ComPtr<ID3D11Query>, 40u> begins, ends;
      auto disjoint = query(D3D11_QUERY_TIMESTAMP_DISJOINT);
      for (UINT i = 0u; i < 40u; ++i) {
        begins[i] = query(D3D11_QUERY_TIMESTAMP);
        ends[i] = query(D3D11_QUERY_TIMESTAMP);
      }
      std::array<sample_t, 40u> samples {};
      const std::array<UINT, 4u> order = ordinal & 1u ? std::array<UINT, 4u> {1u, 0u, 0u, 1u} : std::array<UINT, 4u> {0u, 1u, 1u, 0u};
      const auto measured_begin = clock_t::now();
      context_->Begin(disjoint.Get());
      for (UINT i = 0u; i < 40u; ++i) {
        auto &s = samples[i];
        s.strategy = order[i / 10u];
        s.block = i / 10u;
        s.ordinal = i % 10u;
        context_->End(begins[i].Get());
        const auto start = clock_t::now();
        try {
          s.submitted = submit(s.strategy);
        } catch (const std::exception &e) {
          s.submitted = false;
          s.error = e.what();
        }
        const auto end = clock_t::now();
        s.submit_ns = std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        context_->End(ends[i].Get());
        if (!s.submitted && s.error.empty()) {
          s.error = error_;  // Preserve failure; never remove/retry its sample.
        }
      }
      context_->End(disjoint.Get());
      const auto measured_end = clock_t::now();
      context_->Flush();
      const auto deadline = clock_t::now() + std::chrono::seconds(60);
      D3D11_QUERY_DATA_TIMESTAMP_DISJOINT timing {};
      HRESULT disjoint_status = S_FALSE;
      const bool timing_ready = get_query(disjoint.Get(), timing, deadline, disjoint_status);
      bool samples_ok = true;
      for (UINT i = 0u; i < 40u; ++i) {
        auto &s = samples[i];
        s.start_ready = get_query(begins[i].Get(), s.start_tick, deadline, s.start_status);
        s.end_ready = get_query(ends[i].Get(), s.end_tick, deadline, s.end_status);
        samples_ok = samples_ok && s.submitted && s.start_ready && s.end_ready && s.end_tick >= s.start_tick;
        row["samples"].push_back({{"strategy", s.strategy}, {"block", s.block}, {"ordinal", s.ordinal}, {"submitted", s.submitted}, {"error", s.error}, {"start_ready", s.start_ready}, {"end_ready", s.end_ready}, {"start_hresult", std::int64_t(s.start_status)}, {"end_hresult", std::int64_t(s.end_status)}, {"start_tick", s.start_tick}, {"end_tick", s.end_tick}, {"CPU_submit_ns", s.submit_ns}});
      }
      const auto queried = clock_t::now();
      const auto after = counters();
      const bool cached = cached_delta(before, after, hold ? 27u : 4u);
      row["blocks"] = order;
      row["block_repeats"] = 10u;
      row["frequency"] = timing.Frequency;
      row["disjoint"] = timing.Disjoint != FALSE;
      row["disjoint_ready"] = timing_ready;
      row["disjoint_hresult"] = std::int64_t(disjoint_status);
      row["counters_before"] = before;
      row["counters_after"] = after;
      row["all_measured_cached_recorded"] = cached;
      row["warm_hashes"] = hashes(warm);
      row["direct_reference_hashes"] = hold ? nlohmann::json(nullptr) : hashes(direct);
      bool exact = false;
      try {
        final_capture = capture();
        exact = matches(final_capture, warm) && matches(warm, final_capture);
        row["artifacts"] = save_arrays(folder, final_capture);
      } catch (const std::exception &e) {
        row["collection_failure"] = e.what();
      }
      row["functional_end_exact"] = exact;
      row["status"] = samples_ok && timing_ready && !timing.Disjoint && timing.Frequency > 0u && cached && exact && !row.contains("collection_failure") ? "PASS_PHASE" : "FAIL_PHASE_PRESERVED";
      const auto ns = [&](auto t) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(t - phase_start).count();
      };
      row["phase_ns_from_start"] = {{"measured_begin", ns(measured_begin)}, {"measured_submission_end", ns(measured_end)}, {"queries_read", ns(queried)}};
      row["no_diagnostic_Map_CopyResource_query_read_hash_or_file_in_timestamp_intervals"] = true;
      row["baseline_dispatch_command_calls"] = 458u;
      row["enabled_dispatch_command_calls"] = 732u;
      row["CPU_submit_scope"] = "NATIVE_TEST_ACTUAL_COMPILE_FLAGS_REQUIRED_NOT_LIVE";
      return row;
    }

    void cost() {
      const std::filesystem::path input = std::getenv("SUNSHINE_TEST_OUTLINE_CLOSURE_CASES"), output = std::getenv("SUNSHINE_TEST_OUTLINE_CLOSURE_COST_OUTPUT");
      const auto input_bytes = file_bytes(input);
      const auto spec = nlohmann::json::parse(input_bytes.begin(), input_bytes.end());
      ASSERT_EQ(spec.at("schema"), 2u);
      ASSERT_EQ(spec.at("driver"), "hardware");
      ASSERT_TRUE(hardware_);
      ASSERT_EQ(spec.at("cases").size(), 28u);
      ASSERT_EQ(spec.at("stress").size(), 2u);
      ASSERT_EQ(spec.at("outline_source_closure_sha256"), ring_closure_);
      ASSERT_FALSE(std::filesystem::exists(output));
      ASSERT_TRUE(std::filesystem::create_directories(output));
      std::array<wchar_t, 32768u> path {};
      const auto len = GetModuleFileNameW(nullptr, path.data(), DWORD(path.size()));
      ASSERT_GT(len, 0u);
      ASSERT_LT(len, path.size());
      const std::filesystem::path exe(std::wstring(path.data(), len));
      const auto eb = file_bytes(exe);
      nlohmann::json report {{"schema", 1u}, {"driver", "hardware"}, {"adapter", adapter_}, {"test_executable", {{"path", exe.string()}, {"sha256", hash_bytes(eb)}, {"bytes", eb.size()}}}, {"input_specification", {{"path", input.string()}, {"sha256", hash_bytes(input_bytes)}, {"bytes", input_bytes.size()}}}, {"outline_helper_source_pin", compiled_ring_helper_pin}, {"baseline_helper_source_pin", compiled_baseline_helper_pin}, {"outline_source_closure_sha256", ring_closure_}, {"enclosure_source_closure_sha256", enclosure_closure_}, {"inner_source_closure_sha256", inner_closure_}, {"component_source_closure_sha256", graph_closure_}, {"actual_case_count", 28u}, {"authored_stress_count", 2u}, {"modes", {"active", "root0_hold"}}, {"total_planned_raw_samples", 2400u}, {"all_semantic_status_UNKNOWN", true}, {"qualified_performance_comparison", false}, {"competing_load_permitted", true}, {"cases", nlohmann::json::array()}};
      const auto save = [&] {
        std::ofstream f(output / "outline_support_cost.json", std::ios::binary | std::ios::trunc);
        f << report.dump(2) << '\n';
        require(f.good());
      };
      for (UINT i = 0u; i < 30u; ++i) {
        const auto item = i < 28u ? spec.at("cases").at(i) : spec.at("stress").at(i - 28u);
        nlohmann::json row {{"ordinal", i}, {"name", item.at("name")}, {"authored_stress", i >= 28u}, {"case", item}, {"phases", nlohmann::json::array()}};
        try {
          expected_.clear();
          if (i < 28u) {
            load_saved(item);
          } else {
            load_stress(i - 27u);
          }
          for (const auto &[key, value] : item.at("expected_active").items()) {
            expected_[key] = bound_bytes(value, value.contains("bytes") ? value.at("bytes").get<std::size_t>() : value.at("byte_count").get<std::size_t>());
          }
          row["shape"] = {W_, H_};
          row["source_model_identity_unchanged"] = true;
          row["empty_ocr_record"] = words(read(ocr_.resource.Get()))[4u] == 0u;
          auto held = words(literal_root_);
          held[60u] = 0u;
          context_->UpdateSubresource(root_.resource.Get(), 0u, nullptr, held.data(), 0u, 0u);
          require(submit(0u));
          require(submit(1u));
          require(body() == std::vector<char>(std::size_t(W_) * H_ * 4u, 0) && outline_matte() == std::vector<char>(std::size_t(W_) * H_ * 4u, 0));
          context_->UpdateSubresource(root_.resource.Get(), 0u, nullptr, literal_root_.data(), 0u, 0u);
          const auto direct = direct_control();
          const auto folder = output / ("case_" + std::to_string(i + 1u));
          require(std::filesystem::create_directory(folder));
          arrays_t active;
          auto a = phase("active", i, i >= 28u ? i - 27u : 0u, item, direct, {}, folder / "active", active);
          row["phases"].push_back(a);
          if (a.at("status") != "PASS_PHASE") {
            report["cases"].push_back(row);
            save();
            ADD_FAILURE() << "Active phase failed; all available samples preserved";
            return;
          }
          arrays_t hold;
          auto z = phase("root0_hold", i, i >= 28u ? i - 27u : 0u, item, {}, expected_hold(active), folder / "root0_hold", hold);
          row["phases"].push_back(z);
          report["cases"].push_back(row);
          save();
          if (z.at("status") != "PASS_PHASE") {
            ADD_FAILURE() << "Root0 phase failed; all available samples preserved";
            return;
          }
        } catch (const std::exception &e) {
          row["setup_or_collection_failure"] = e.what();
          report["cases"].push_back(row);
          save();
          ADD_FAILURE() << e.what();
          return;
        }
      }
      report["status"] = "COMPLETE_ALL30_BOTH_MODES_ALL2400_SAMPLES";
      save();
    }

    baseline::resources_t literal_;
    std::vector<char> literal_root_;
    arrays_t expected_;
  };

  TEST_F(HostSbsCurrentOutlineSupportCostGpuTest, Original30ActiveAndRoot0PairedWholeHelperCost) {
    cost();
  }
}  // namespace
