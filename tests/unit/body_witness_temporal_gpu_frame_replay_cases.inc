// Additional opt-in proof only. Existing tests and default/null program remain unchanged.
  TEST_F(HostSbsBodyWitnessTemporalGpuTest, SavedGpuFrameContextWholeBackendReplay) {
    if (spec_path_.empty()) {
      GTEST_SKIP() << "Explicit saved sequence fixture required";
    }
    ASSERT_EQ(input_spec_.at("schema").get<UINT>(), 1u);
    ASSERT_TRUE(input_spec_.at("gpu_frame_replay").get<bool>());
    namespace frame_context = models::host_sbs_temporal_frame_context_gpu;
    namespace cache = models::host_sbs_shader_cache;
    const std::array context_specs {cache::shader_spec {"host_sbs_temporal_frame_context_cs.hlsl", "main", "cs_5_0"}};
    const auto context_sources = cache::snapshot_sources(input_spec_.at("frame_context_root").get<std::string>(), context_specs);
    ASSERT_TRUE(context_sources);
    ASSERT_EQ(cache::source_closure_sha256(context_sources), input_spec_.at("frame_context_source_closure_sha256").get<std::string>());
    const auto context_code = cache::get(context_sources, context_specs[0]);
    ASSERT_TRUE(context_code && !context_code->empty());
    ComPtr<ID3D11ComputeShader> context_shader;
    ASSERT_TRUE(SUCCEEDED(device_->CreateComputeShader(context_code->data(), context_code->size(), nullptr, &context_shader)));
    frame_context::resources_t context_transport;
    ASSERT_TRUE(context_transport.ensure(device_.Get(), error_)) << error_;
    colour_buffer_t context_uip;
    ASSERT_TRUE(colour_buffer(device_.Get(), 40u, false, context_uip));
    ComPtr<ID3D11Buffer> context_root;
    ComPtr<ID3D11ShaderResourceView> context_root_srv;
    D3D11_BUFFER_DESC root_desc {};
    root_desc.ByteWidth = 256u; root_desc.Usage = D3D11_USAGE_DEFAULT;
    root_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    root_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    ASSERT_TRUE(SUCCEEDED(device_->CreateBuffer(&root_desc, nullptr, &context_root)));
    D3D11_SHADER_RESOURCE_VIEW_DESC root_view {};
    root_view.Format = DXGI_FORMAT_R32_TYPELESS; root_view.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
    root_view.BufferEx.NumElements = 64u; root_view.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
    ASSERT_TRUE(SUCCEEDED(device_->CreateShaderResourceView(context_root.Get(), &root_view, &context_root_srv)));

    const char *out = std::getenv("SUNSHINE_TEST_BODY_WITNESS_TEMPORAL_OUTPUT");
    ASSERT_NE(out, nullptr);
    const std::filesystem::path output = out;
    ASSERT_FALSE(std::filesystem::exists(output));
    std::filesystem::create_directories(output);
    nlohmann::json aggregate {{"schema", 1}, {"scope", "GPU_CONTEXT_TO_RECORDED_BACKEND_STRICT_ORIGINAL_V2_REFERENCE"}, {"cases", nlohmann::json::array()}};
    const auto save = [&](const std::filesystem::path &p, const std::vector<char> &bytes) {
      std::filesystem::create_directories(p.parent_path());
      std::ofstream stream(p, std::ios::binary);
      stream.write(bytes.data(), std::streamsize(bytes.size()));
      if (!stream) {
        throw std::runtime_error("Cannot save temporal readback");
      }
      return nlohmann::json {{"path", p.string()}, {"sha256", colour_sha(bytes.data(), bytes.size())}, {"bytes", bytes.size()}};
    };
    UINT observation = 0u;
    for (const auto &sequence : input_spec_.at("sequences")) {
      backend_ = temporal::resources_t {};  // Explicit independent authenticated sequence scope, no serial compression.
      for (const auto &item : sequence.at("cases")) {
        const auto &a = item.at("artifacts");
        auto cbytes = colour_bound(a.at("constants"), 96u);
        std::memcpy(constants_.words.data(), cbytes.data(), 96u);
        const auto c = constants_.words;
        if (width_ != c[0] || height_ != c[1] || crop_top_ != c[2] || crop_height_ != c[3]) {
          ASSERT_TRUE(resize(c[0], c[1], c[2], c[3]));
        }
        std::memcpy(constants_.words.data(), cbytes.data(), 96u);
        source_.resize(std::size_t(width_) * crop_height_);
        auto raw = colour_bound(a.at("source_band"), source_.size() * 8u);
        std::memcpy(source_.data(), raw.data(), raw.size());
        probability_.resize(153600u);
        raw = colour_bound(a.at("probability"), 153600u * 4u);
        std::memcpy(probability_.data(), raw.data(), raw.size());
        support_ = colour_words(colour_bound(a.at("current_support"), source_.size() * 4u));
        ocr_ = colour_words(colour_bound(a.at("ocr_record"), 1024u));
        raw = colour_bound(a.at("frame_constants"), 256u);
        std::memcpy(frame_.words.data(), raw.data(), 256u);
        const auto root = colour_words(colour_bound(a.at("root_arguments"), 12u));
        ASSERT_EQ(root[1], 1u);
        ASSERT_EQ(root[2], 1u);
        ASSERT_LE(root[0], 1u);
        root_gate_ = root[0];
        ASSERT_EQ(frame_.words[48], c[12]);
        ASSERT_EQ(frame_.words[49], c[13]);
        ASSERT_EQ(frame_.words[50], c[16]);
        ASSERT_EQ(frame_.words[51], c[17]);
        ASSERT_TRUE(backend_.ensure(device_.Get(), width_, crop_height_, capacities_, error_));
        const auto execution = input_spec_.value("execution", std::string("recorded"));
        ASSERT_TRUE(execution == "recorded" || execution == "direct");
        ASSERT_EQ(execution, "recorded");
        const auto &transport_case = item.at("frame_transport");
        const auto uip_bytes = colour_bound(transport_case.at("uip"), 160u);
        const auto full_root_bytes = colour_bound(transport_case.at("transaction"), 256u);
        const auto seed_bytes = colour_bound(transport_case.at("seed"), 256u);
        EXPECT_EQ(std::vector<char>(full_root_bytes.begin() + 240u, full_root_bytes.begin() + 252u), colour_bound(a.at("root_arguments"), 12u));
        frame_context::frame_t seed;
        std::memcpy(seed.words.data(), seed_bytes.data(), 256u);
        // Literal current context only. Expected masks/native references are read AFTER dispatch.
        context_->UpdateSubresource(constants_buffer_.Get(), 0u, nullptr, cbytes.data(), 0u, 0u);
        context_->UpdateSubresource(context_uip.value.Get(), 0u, nullptr, uip_bytes.data(), 0u, 0u);
        context_->UpdateSubresource(context_root.Get(), 0u, nullptr, full_root_bytes.data(), 0u, 0u);
        const auto context_cb_pending = archive(constants_buffer_.Get());
        const auto context_uip_pending = archive(context_uip.value.Get());
        const auto context_root_pending = archive(context_root.Get());
        ASSERT_TRUE(context_transport.queue(context_.Get(), context_shader.Get(), constants_buffer_.Get(), context_uip.srv.Get(), context_root_srv.Get(), seed, error_)) << error_;
        const auto serialized_pending = archive(context_transport.output().Get());
        // NO Map, GetData, CPU cut decode or result-dependent dispatch exists between these calls.
        ASSERT_TRUE(run_composed(true, context_transport.output().Get())) << error_;
        const auto serialized = read_archive(serialized_pending);
        const auto uploaded_context_cb = read_archive(context_cb_pending);
        const auto uploaded_context_uip = read_archive(context_uip_pending);
        const auto uploaded_context_root = read_archive(context_root_pending);
        EXPECT_EQ(uploaded_context_cb, cbytes);
        EXPECT_EQ(uploaded_context_uip, uip_bytes);
        EXPECT_EQ(uploaded_context_root, full_root_bytes);
        EXPECT_EQ(serialized, colour_bound(a.at("frame_constants"), 256u));
        expect_uploaded_exact();
        EXPECT_EQ(temporal_uploaded_.at("uploaded_frame_constants.u32"), colour_raw(frame_.words));
        EXPECT_EQ(temporal_uploaded_.at("uploaded_root_arguments.u32"), colour_bound(a.at("root_arguments"), 12u));
        if (root_gate_ == 1u) {
          expect_available();
        } else {
          EXPECT_TRUE(std::all_of(temporal_receipt_.begin(), temporal_receipt_.end(), [](UINT v) {
            return v == 0u;
          }));
          EXPECT_TRUE(std::all_of(temporal_body_.begin(), temporal_body_.end(), [](UINT v) {
            return v == 0u;
          }));
          EXPECT_TRUE(std::all_of(temporal_final_.begin(), temporal_final_.end(), [](UINT v) {
            return v == 0u;
          }));
        }
        const auto folder = output / ("case_" + std::to_string(++observation));
        nlohmann::json artifacts;
        for (const auto &[name, bytes] : uploaded_) {
          artifacts[name] = save(folder / name, bytes);
        }
        for (const auto &[name, bytes] : temporal_uploaded_) {
          artifacts[name] = save(folder / name, bytes);
        }
        artifacts["uploaded_context_constants.u32"] = save(folder / "uploaded_context_constants.u32", uploaded_context_cb);
        artifacts["uploaded_context_uip.u32"] = save(folder / "uploaded_context_uip.u32", uploaded_context_uip);
        artifacts["uploaded_context_root.u32"] = save(folder / "uploaded_context_root.u32", uploaded_context_root);
        artifacts["declared_context_seed.u32"] = save(folder / "declared_context_seed.u32", seed_bytes);
        artifacts["serialized_frame256.u32"] = save(folder / "serialized_frame256.u32", serialized);
        artifacts["baseline_body.u32"] = save(folder / "baseline_body.u32", colour_raw(body_));
        artifacts["baseline_final.u32"] = save(folder / "baseline_final.u32", colour_raw(final_));
        bool exact = true;
        for (const auto &[name, bytes] : std::array<std::pair<const char *, std::vector<char>>, 4u> {{{"body", colour_raw(temporal_body_)}, {"final", colour_raw(temporal_final_)}, {"baseline_body", colour_raw(body_)}, {"baseline_final", colour_raw(final_)}}}) {
          exact &= bytes == colour_bound(item.at("expected").at(name), source_.size() * 4u);
        }
        bool original_native_exact = true;
        for (const char *name : {"body.u32", "final.u32", "receipt.u32", "state.u32", "uploaded_frame_constants.u32"}) {
          original_native_exact &= temporal_uploaded_.at(name) == colour_bound(transport_case.at("original_native_reference").at(name), temporal_uploaded_.at(name).size());
        }
        EXPECT_TRUE(original_native_exact) << "Original same-driver V2 full body/M/receipt/state/Frame256 must remain exact";
        const bool context_exact = uploaded_context_cb == cbytes && uploaded_context_uip == uip_bytes && uploaded_context_root == full_root_bytes && serialized == colour_bound(a.at("frame_constants"), 256u);
        const nlohmann::json identity {{"gpu_frame_transport", true}, {"original_native_exact", original_native_exact}, {"context_operands_exact", context_exact},
          {"seed_capture_scope", "DECLARED_HOST_CONSTANT_SOURCE_USED_BY_HELPER_NOT_GPU_READBACK"},
          {"no_Map_between_GPU_serializer_and_backend", true}, {"frame_context_source_closure_sha256", input_spec_.at("frame_context_source_closure_sha256")}, {"schema", 1}, {"case", item}, {"sequence", sequence.at("name")}, {"driver", driver_}, {"adapter", adapter_identity()}, {"temporal_source_closure_sha256", temporal_closure_}, {"inner_source_closure_sha256", inner_closure_}, {"component_source_closure_sha256", component_closure_}, {"artifacts", artifacts}, {"expected_whole_baseline_and_candidate_byte_exact", exact}, {"all_semantic_status_UNKNOWN", true}, {"dispatch_policy", "GPU context fixed1 -> fullGPUFrame copy outsidecachedlist -> one recorded original validate/palette/emission/4CC+temporal pipeline. No Map between serializer/backend; no CPU cut or colour/mask decisions."}, {"execution", execution}, {"recordings", coordinator_.diagnostics().recordings}, {"replays", coordinator_.diagnostics().replays}, {"direct_emissions", coordinator_.diagnostics().direct_emissions}, {"source_identity_scope", item.at("provenance")}};
        std::filesystem::create_directories(folder);
        std::ofstream file(folder / "body_witness_temporal_identity.json");
        file << identity.dump(2) << '\n';
        ASSERT_TRUE(file.good());
        aggregate["cases"].push_back(identity);
        EXPECT_TRUE(exact) << "Saved output preserved; no metric or rule waiver";
      }
    }
    std::ofstream file(output / "body_witness_temporal_gpu_frame_replay_identity.json");
    file << aggregate.dump(2) << '\n';
    ASSERT_TRUE(file.good());
  }
