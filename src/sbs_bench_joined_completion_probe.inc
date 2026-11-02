// Included inside sbs_bench's private anonymous namespace AFTER parse_opts/helpers.
// Separate default-off path. It never enters or modifies the ordinary fixed-frame loop.
    nlohmann::json jp_file(const fs::path &p) {
      const auto digest=models::file_sha256_hex(p);
      if(digest.empty()) throw std::runtime_error("Probe artifact hash failed: "+p.string());
      return {{"path",fs::absolute(p).lexically_normal().string()},{"sha256",digest},{"bytes",fs::file_size(p)}};
    }
    void jp_json(const fs::path &p,const nlohmann::json &j) {
      if(fs::exists(p)) throw std::runtime_error("Probe refuses to overwrite: "+p.string());
      std::ofstream f(p);f<<j.dump(2)<<'\n';f.flush();
      if(!f.good()) throw std::runtime_error("Probe JSON write failed");
    }
    std::string jp_phase(external_actual_async_caption_probe::phase_e p) {
      using phase=external_actual_async_caption_probe::phase_e;
      switch(p) {
        case phase::matched_A:return "matched_A";
        case phase::busy_B_current_colour:return "busy_B";
        case phase::busy_empty_current_colour:return "busy_empty";
        case phase::late_B_geometry_current_empty:return "late_B_geometry_current_empty";
        case phase::matched_empty_C:return "matched_empty_C";
      }
      throw std::runtime_error("Unknown probe phase");
    }
    nlohmann::json jp_gate(const std::optional<models::joined_completion_gate_benchmark::snapshot_t> &value) {
      if(!value) return nullptr;
      const auto &v=*value;
      return {{"target_frame",v.configuration.target_frame_id},{"queued",v.queued},{"entered",v.callback_entered},{"exited",v.callback_exited},
        {"released",v.released},{"cancelled",v.cancelled},{"callback_timeout",v.callback_timed_out},{"entry_timeout",v.callback_entry_timed_out},
        {"capture_checked",v.capture_state_checked},{"capture_query_result",v.capture_state_query_result},{"capture_state",v.capture_state},
        {"enqueue_result",v.enqueue_result},{"actual_event_queries",v.actual_event_queries},{"actual_not_ready_queries",v.actual_not_ready_queries},
        {"actual_ready_queries",v.actual_ready_queries},{"actual_failed_queries",v.actual_failed_queries},{"last_actual_event_query",v.last_actual_event_query},
        {"entered_benchmark_wall_us",v.entered_wall_us},{"released_benchmark_wall_us",v.released_wall_us},{"exited_benchmark_wall_us",v.exited_wall_us},
        {"walltime_is_not_source_or_observation_clock",true}};
    }
    nlohmann::json jp_publication(const models::host_sbs_gpu_completion_receipt::expected_t &p) {
      return {{"estimator_generation",p.estimator_generation},{"publication_sequence",p.publication_sequence},
        {"analysis_generation",p.analysis_generation},{"frame_id",p.frame_id},{"transaction_token",p.transaction_token},
        {"domain_tag",p.domain_tag},{"observation_timestamp_us",p.observation_timestamp_us},{"baseline_frame_id",p.baseline_frame_id},
        {"width",p.width},{"height",p.height},{"subtitle_domain",p.subtitle_domain},{"expected_work",p.expected_work},
        {"submission_class",static_cast<unsigned>(p.submission_class)},{"flags",p.flags},{"joint_plane_mode",p.joint_plane_mode},
        {"host_subtitle_outcome",static_cast<unsigned>(p.host_subtitle_outcome)},{"raw_coordinate_scale",p.raw_coordinate_scale}};
    }
    struct jp_texture_t {
      ComPtr<ID3D11Texture2D> texture;
      ComPtr<ID3D11ShaderResourceView> srv;
      ComPtr<ID3D11UnorderedAccessView> uav;
    };
    void jp_require(HRESULT h,const char *what) { if(FAILED(h)) throw std::runtime_error(what); }
    jp_texture_t jp_texture(ID3D11Device *dev,UINT w,UINT h,DXGI_FORMAT format,UINT flags,const void *pixels=nullptr,UINT pitch=0) {
      jp_texture_t out;D3D11_TEXTURE2D_DESC d{};d.Width=w;d.Height=h;d.MipLevels=1;d.ArraySize=1;d.Format=format;d.SampleDesc.Count=1;d.Usage=D3D11_USAGE_DEFAULT;d.BindFlags=flags;
      D3D11_SUBRESOURCE_DATA data{pixels,pitch,0};jp_require(dev->CreateTexture2D(&d,pixels?&data:nullptr,&out.texture),"Probe texture allocation failed");
      if(flags&D3D11_BIND_SHADER_RESOURCE) jp_require(dev->CreateShaderResourceView(out.texture.Get(),nullptr,&out.srv),"Probe SRV failed");
      if(flags&D3D11_BIND_UNORDERED_ACCESS) jp_require(dev->CreateUnorderedAccessView(out.texture.Get(),nullptr,&out.uav),"Probe UAV failed");
      return out;
    }
    D3D11_TEXTURE2D_DESC jp_desc(ID3D11ShaderResourceView *view) {
      if(!view) throw std::runtime_error("Probe missing texture view");
      ComPtr<ID3D11Resource> r;ComPtr<ID3D11Texture2D> t;view->GetResource(&r);jp_require(r.As(&t),"Probe expected texture");D3D11_TEXTURE2D_DESC d{};t->GetDesc(&d);return d;
    }
    // One stage per layout, independent immutable CPU bytes per phase. No Map
    // destination aliases an estimator resource or another phase's vector.
    struct jp_stage_t {
      ComPtr<ID3D11Resource> resource;
      D3D11_RESOURCE_DIMENSION dimension=D3D11_RESOURCE_DIMENSION_UNKNOWN;
      D3D11_TEXTURE2D_DESC texture{};
      UINT buffer_bytes=0;
      std::size_t row_bytes=0,rows=0;
    };
    struct jp_capture_t {
      std::array<std::vector<char>,60> bytes;
      std::array<bool,60> copied{};
      bool readback_complete=false;
      std::int64_t readback_complete_wall_us=0;
      std::optional<external_actual_async_caption_probe::phase_record_t> authenticated_row;
      nlohmann::json metadata;
    };
    #include "sbs_bench_pre_unmap_feasibility.inc"
    #include "sbs_bench_queued_same_source_probe.inc"
    int run_joined_completion_probe(const opts &o,const std::vector<fs::path> &frames,
      const std::vector<std::uint64_t> &timestamps,ID3D11Device *dev,ID3D11DeviceContext *ctx,
      models::video_depth_estimator &est,ID3D11VertexShader *vs,ID3D11PixelShader *ps,ID3D11SamplerState *sampler,int max_width) {
      namespace probe=external_actual_async_caption_probe;
      namespace held=external_held_geometry_current_colour_v2;
      auto root=fs::path(o.out)/"joined-completion-probe-v4";
      bool created=false;
      std::string deferred_error;
      try {
        std::function<void()> flush_deferred;
        if(frames.size()!=3||timestamps.size()!=3||!vs||!ps||!sampler) throw std::runtime_error("Probe requires exactly3 sources and a new output folder");
        std::string plan_bytes;if(!read_file_snapshot(o.joined_completion_probe,plan_bytes)) throw std::runtime_error("Cannot snapshot probe plan");
        const auto plan=nlohmann::json::parse(plan_bytes);
        if(plan.value("probe_kind",std::string{})=="queued-same-source-draw-v1")
          return run_queued_same_source_probe(o,frames,timestamps,dev,ctx,est,vs,ps,sampler,max_width,plan_bytes);
        if(plan.value("probe_kind",std::string{})=="actual-pre-unmap-feasibility-v1")
          return run_pre_unmap_feasibility(o,frames,timestamps,dev,ctx,est,plan_bytes);
        if(plan.at("schema")!=3||plan.at("presentation_policy")!="bounded-approximate-unknown-cut"||plan.at("sources").size()!=3||
           plan.at("observation_timestamps_us")!=nlohmann::json(timestamps)) throw std::runtime_error("Invalid immutable probe plan/timeline");
        if(plan.at("estimator_frame_ids")!=nlohmann::json::array({2,3,4}) ||
           plan.at("source_conversion_ids")!=nlohmann::json::array({1,2,3}) ||
           plan.at("prewarm").at("estimator_frame_id")!=1 ||
           plan.at("prewarm").at("observation_timestamp_us")!=1 ||
           plan.at("prewarm").at("source_index")!=2 ||
           plan.at("hold_after_entry_ms")!=120 || plan.at("empty_after_entry_ms")!=33)
          throw std::runtime_error("Invalid explicit prewarm/identity/gate contract");
        if(plan.at("typed_presentation_policy").at("version")!=2 ||
           plan.at("typed_presentation_policy").at("source_sha256")!="7eb1e2b8c63ce042f9426788a99becd8dbd50cc949d306e9363ebd0dead5c2cf")
          throw std::runtime_error("Invalid V4 typed presentation policy identity");
        // OPTIONAL transport-only extension. Missing/disabled section runs the V4 draw path
        // with the original twelve exports and metadata. No new GPU resources/commands then.
        namespace snapshot=models::research_current_source_snapshot_v1;
        const bool snapshot_enabled=plan.contains("current_source_snapshot") &&
          plan.at("current_source_snapshot").value("enabled",false);
        snapshot::resources_t current_snapshot;
        std::array<ComPtr<ID3D11ComputeShader>,3> snapshot_shaders;
        std::uint64_t snapshot_epoch_us=0,snapshot_permission_id=0,snapshot_analysis_generation=0,snapshot_domain_tag=0;
        UINT snapshot_band_height=0,snapshot_field_width=0,snapshot_field_height=0;
        std::array<std::uint32_t,4> snapshot_content{};
        std::string snapshot_closure;
        nlohmann::json snapshot_plan;
        namespace cuepipe=models::research_current_cue_pipeline_v6;
        const bool cue_enabled=plan.contains("native_current_rgb_cue")&&plan.at("native_current_rgb_cue").value("enabled",false);
        if(cue_enabled&&!snapshot_enabled)throw std::runtime_error("V6 cue requires own current-source snapshot");
        cuepipe::pipeline_t cue_pipeline;
        std::array<ComPtr<ID3D11ComputeShader>,20> cue_shaders;
        std::array<ComPtr<ID3D11ComputeShader>,3> cue_graph_shaders;
        ComPtr<ID3D11ComputeShader> cue_context_wire;
        nlohmann::json cue_plan,frozen_A_hashes;
        bool cue_getter_mismatch_guard=false;
        namespace retirement=models::research_source_pattern_retirement_v1;
        const bool retirement_enabled=plan.contains("source_pattern_retirement")&&plan.at("source_pattern_retirement").value("enabled",false);
        if(retirement_enabled&&(!cue_enabled||!snapshot_enabled))throw std::runtime_error("V7 retirement requires literal V6 source/cue setup");
        retirement::resources_t old_group_retirement;
        std::array<ComPtr<ID3D11ComputeShader>,8> retirement_shaders;
        ComPtr<ID3D11PixelShader> retirement_ps;
        nlohmann::json retirement_plan;
        std::string retirement_closure,retirement_render_closure;


        if(snapshot_enabled) {
          snapshot_plan=plan.at("current_source_snapshot");
          if(snapshot_plan.at("schema")!=1 || snapshot_plan.at("header").at("sha256").get<std::string>()!="a88c522e1fd0cf9741a9e6474de801baac6db36d96ddd40b8f52e79e7645d14c" ||
             snapshot_plan.at("permission_kind")!="private-authored-three-source-immutable-owner" ||
             snapshot_plan.at("source_clock_kind")!="actual-authored-capture-steady-us" ||
             snapshot_plan.at("semantic_authority")!=false)
            throw std::runtime_error("Invalid private current-source transport plan");
          root=fs::path(o.out)/"joined-completion-probe-v5";
          constexpr std::array<models::host_sbs_shader_cache::shader_spec,3> snapshot_specs {{{"host_sbs_current_source_snapshot_cs.hlsl","clear_main","cs_5_0"},{"host_sbs_current_source_snapshot_cs.hlsl","snapshot_main","cs_5_0"},{"host_sbs_current_source_snapshot_cs.hlsl","resolve_main","cs_5_0"}}};
          const auto snapshot_sources=models::host_sbs_shader_cache::snapshot_sources(snapshot_plan.at("shader_root").get<std::string>(),snapshot_specs);
          snapshot_closure=models::host_sbs_shader_cache::source_closure_sha256(snapshot_sources);
          if(!snapshot_sources || snapshot_closure!=snapshot_plan.at("source_closure_sha256").get<std::string>() || snapshot_closure!="0dca5e8bc18c1795fadaf93a9fbd4b7a6d3d665ec310708f6b78b18235c78244")
            throw std::runtime_error("Current-source transport shader closure mismatch");
          for(std::size_t i=0;i<snapshot_specs.size();++i) {
            const auto code=models::host_sbs_shader_cache::get(snapshot_sources,snapshot_specs[i]);
            if(!code||code->empty())throw std::runtime_error("Current-source transport shader compile unavailable");
            jp_require(dev->CreateComputeShader(code->data(),code->size(),nullptr,&snapshot_shaders[i]),"Current-source transport CS creation failed");
          }
        }
        if(cue_enabled) {
          cue_plan=plan.at("native_current_rgb_cue");
          if(cue_plan.at("schema")!=1||cue_plan.at("prior_policy")!="genuine-completed-A-only;matched-A-canonical-no-prior"||
             cue_plan.at("pipeline_header").at("sha256").get<std::string>()!="8770c837d518ad860266e03984ee4f7f935c5079824e3edcfa059f5963157d09"||
             cue_plan.at("helper_source_sha256")!="2cbdd4545c1d2c3829e8e3dbc5cb0d82cb4021be161b972d6e1e1aa8862fc9c8"||cue_plan.at("semantic_authority")!=false)
            throw std::runtime_error("Invalid V6 source-to-cue research plan");
          if(!est.set_completed_ocr_conditioner_research_enabled(true))throw std::runtime_error("V6 completed-A diagnostic flag must precede first input");
          root=fs::path(o.out)/"joined-completion-probe-v6";
          std::vector<models::host_sbs_shader_cache::shader_spec> cue_specs;
          for(const auto *entry:models::host_sbs_current_rgb_cue_gpu_v1::cue_entries)cue_specs.push_back({"host_sbs_current_rgb_cue_cs.hlsl",entry,"cs_5_0"});
          const std::array graph_specs {
            models::host_sbs_shader_cache::shader_spec{"host_sbs_subtitle_seeded_components_cs.hlsl","init_main","cs_5_0"},
            models::host_sbs_shader_cache::shader_spec{"host_sbs_subtitle_seeded_components_cs.hlsl","union_main","cs_5_0"},
            models::host_sbs_shader_cache::shader_spec{"host_sbs_subtitle_seeded_components_cs.hlsl","reduce_main","cs_5_0"},
            models::host_sbs_shader_cache::shader_spec{"host_sbs_subtitle_seeded_components_cs.hlsl","resolve_main","cs_5_0"},
            models::host_sbs_shader_cache::shader_spec{"host_sbs_subtitle_seeded_components_cs.hlsl","dilate_main","cs_5_0"},
            models::host_sbs_shader_cache::shader_spec{"host_sbs_subtitle_seeded_components_cs.hlsl","dilate_box_main","cs_5_0"}};
          const std::array wire_specs {models::host_sbs_shader_cache::shader_spec{"current_cue_context_from_source_cs.hlsl","context_main","cs_5_0"}};
          const auto cue_sources=models::host_sbs_shader_cache::snapshot_sources(cue_plan.at("cue_shader_root").get<std::string>(),cue_specs);
          const auto graph_sources=models::host_sbs_shader_cache::snapshot_sources(cue_plan.at("component_shader_root").get<std::string>(),graph_specs);
          const auto wire_sources=models::host_sbs_shader_cache::snapshot_sources(cue_plan.at("context_wire_shader_root").get<std::string>(),wire_specs);
          if(models::host_sbs_shader_cache::source_closure_sha256(cue_sources)!="51c945dddd62e2ad558e2037cab4cb3b2b2b541ca9f1f78bc82f6921585872e1"||
             models::host_sbs_shader_cache::source_closure_sha256(graph_sources)!="7e2f44690d64cebe3261133a3642fa558c8c25c7086bf19551f284a4cb13ebe0"||
             models::host_sbs_shader_cache::source_closure_sha256(wire_sources)!="0b0bf1c6a081ca79a7d2ea861a87f175d11027461420676868027a37a4995f29")throw std::runtime_error("V6 immutable shader closure mismatch");
          for(std::size_t i=0;i<cue_specs.size();++i){auto code=models::host_sbs_shader_cache::get(cue_sources,cue_specs[i]);if(!code||code->empty())throw std::runtime_error("V6 cue compile unavailable");jp_require(dev->CreateComputeShader(code->data(),code->size(),nullptr,&cue_shaders[i]),"V6 cue CS failed");}
          for(std::size_t i=0;i<cue_graph_shaders.size();++i){auto code=models::host_sbs_shader_cache::get(graph_sources,graph_specs[i]);if(!code||code->empty())throw std::runtime_error("V6 graph compile unavailable");jp_require(dev->CreateComputeShader(code->data(),code->size(),nullptr,&cue_graph_shaders[i]),"V6 graph CS failed");}
          const auto code=models::host_sbs_shader_cache::get(wire_sources,wire_specs[0]);if(!code||code->empty())throw std::runtime_error("V6 current-context wire compile unavailable");jp_require(dev->CreateComputeShader(code->data(),code->size(),nullptr,&cue_context_wire),"V6 context wire CS failed");
        }

        if(retirement_enabled) {
          retirement_plan=plan.at("source_pattern_retirement");
          if(retirement_plan.at("schema")!=1||retirement_plan.at("rule_freeze_sha256")!="7227a06053bbf1aa057eb66996b21c43f230b2fbc10be28e03569e81e72246f1"||
             retirement_plan.at("negative_only")!=true||retirement_plan.at("producer_semantic_presence")!="UNKNOWN")throw std::runtime_error("Invalid frozen negative-only retirement plan");
          std::vector<models::host_sbs_shader_cache::shader_spec> retirement_specs;
          for(const auto *entry:retirement::entries)retirement_specs.push_back({"host_sbs_source_pattern_retirement_cs.hlsl",entry,"cs_5_0"});
          const auto sources=models::host_sbs_shader_cache::snapshot_sources(retirement_plan.at("shader_root").get<std::string>(),retirement_specs);
          retirement_closure=models::host_sbs_shader_cache::source_closure_sha256(sources);
          if(!sources||retirement_closure!=retirement_plan.at("source_closure_sha256").get<std::string>())throw std::runtime_error("Frozen retirement source closure mismatch");
          for(std::size_t i=0;i<retirement_specs.size();++i){const auto code=models::host_sbs_shader_cache::get(sources,retirement_specs[i]);if(!code||code->empty())throw std::runtime_error("Retirement CS compile unavailable");jp_require(dev->CreateComputeShader(code->data(),code->size(),nullptr,&retirement_shaders[i]),"Retirement CS creation failed");}
          const std::array render_specs{models::host_sbs_shader_cache::shader_spec{"sbs_reprojection_v2_live_ps.hlsl","main_ps","ps_5_0"}};
          const auto render_sources=models::host_sbs_shader_cache::snapshot_sources(retirement_plan.at("private_render_shader_root").get<std::string>(),render_specs);
          retirement_render_closure=models::host_sbs_shader_cache::source_closure_sha256(render_sources);
          if(!render_sources||retirement_render_closure!=retirement_plan.at("private_render_source_closure_sha256").get<std::string>())throw std::runtime_error("Private negative PS closure mismatch");
          const auto code=models::host_sbs_shader_cache::get(render_sources,render_specs[0]);if(!code||code->empty())throw std::runtime_error("Private negative PS compile unavailable");jp_require(dev->CreatePixelShader(code->data(),code->size(),nullptr,&retirement_ps),"Private negative PS creation failed");
          root=fs::path(o.out)/"joined-completion-probe-v7";
        }
        if(fs::exists(root)||!fs::create_directory(root))throw std::runtime_error("Probe requires new output folder");
        created=true;
        const auto private_root=fs::path(plan.at("withdraw_shader_root").get<std::string>());
        constexpr std::array<models::host_sbs_shader_cache::shader_spec,1> specs {{{"held_glyph_withdraw_cs.hlsl","withdraw_main","cs_5_0"}}};
        const auto sources=models::host_sbs_shader_cache::snapshot_sources(private_root,specs);
        const auto closure=models::host_sbs_shader_cache::source_closure_sha256(sources);
        if(!sources||closure!=plan.at("withdraw_source_closure_sha256").get<std::string>()) throw std::runtime_error("Private withdrawal closure mismatch");
        const auto bytecode=models::host_sbs_shader_cache::get(sources,specs[0]);ComPtr<ID3D11ComputeShader> withdraw;
        if(!bytecode) throw std::runtime_error("Private withdrawal compile unavailable");
        jp_require(dev->CreateComputeShader(bytecode->data(),bytecode->size(),nullptr,&withdraw),"Private withdrawal CS creation failed");
        std::array<jp_texture_t,3> inputs;std::array<probe::frame_t,3> packets;std::array<std::string,3> upload_hashes;UINT width=0,height=0;
        nlohmann::json input_bindings=nlohmann::json::array();
        constexpr std::array<const char *,3> roles{{"A","B","empty"}};
        for(std::size_t i=0;i<3;++i) {
          const auto &b=plan.at("sources").at(i);
          if(b.at("role")!=roles[i]||fs::weakly_canonical(b.at("path").get<std::string>())!=fs::weakly_canonical(frames[i])) throw std::runtime_error("Probe source-role/path mismatch");
          std::string bytes;rgba_image image;
          if(!read_file_snapshot(frames[i],bytes)||sha256_hex(bytes)!=b.at("sha256").get<std::string>()||bytes.size()!=b.at("bytes").get<std::size_t>()||!load_png(std::string_view(bytes),image)) throw std::runtime_error("Probe source snapshot/hash/PNG mismatch");
          if(!i){width=image.w;height=image.h;}else if(image.w!=width||image.h!=height) throw std::runtime_error("Probe source domain changed");
          if(!width||!height||width>5120||height>5120) throw std::runtime_error("Probe source dimensions invalid");
          inputs[i]=jp_texture(dev,width,height,DXGI_FORMAT_B8G8R8A8_UNORM,D3D11_BIND_SHADER_RESOURCE,image.bgra.data(),width*4);
          packets[i].immutable_full_colour=inputs[i].srv.Get();packets[i].observation_timestamp_us=timestamps[i];packets[i].estimator_frame_id=i+2;packets[i].conversion_serial=i+1;packets[i].source_file_sha256=sha256_hex(bytes);
          upload_hashes[i]=sha256_hex(std::string_view(reinterpret_cast<const char *>(image.bgra.data()),image.bgra.size()));
          auto binding=b;binding["decoded_bgra8_sha256"]=upload_hashes[i];binding["width"]=width;binding["height"]=height;input_bindings.push_back(binding);
        }
        std::string geometry_error;const auto output=detail::resolve_sbs_geometry(width,height,o.eye_w,o.eye_h,o.output_scale,max_width,geometry_error);
        if(!output) throw std::runtime_error("Probe geometry invalid: "+geometry_error);
        auto packed=jp_texture(dev,output->sbs_width,output->sbs_height,DXGI_FORMAT_B8G8R8A8_UNORM,D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE);
        ComPtr<ID3D11RenderTargetView> packed_rtv;jp_require(dev->CreateRenderTargetView(packed.texture.Get(),nullptr,&packed_rtv),"Probe RTV failed");
        jp_texture_t retirement_control;ComPtr<ID3D11RenderTargetView> retirement_control_rtv;
        if(retirement_enabled){retirement_control=jp_texture(dev,output->sbs_width,output->sbs_height,DXGI_FORMAT_B8G8R8A8_UNORM,D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE);jp_require(dev->CreateRenderTargetView(retirement_control.texture.Get(),nullptr,&retirement_control_rtv),"Retirement control RTV failed");}

        const auto geometry=models::make_host_sbs_v2_full_frame_geometry(output->content_scale_x,output->content_scale_y);
        const auto b2=const_buffer(dev,geometry);if(!b2) throw std::runtime_error("Probe geometry CB failed");
        D3D11_VIEWPORT viewport{0,0,float(output->sbs_width),float(output->sbs_height),0,1};
        held::domain_t domain;domain.source_width=width;domain.source_height=height;domain.format=DXGI_FORMAT_B8G8R8A8_UNORM;domain.mip_levels=1;domain.array_size=1;domain.sample_count=1;domain.analysis_ltrb={0,0,width,height};
        std::array<jp_texture_t,2> alive;int alive_index=-1;std::uint64_t alive_publication=0,alive_generation=0;
        constexpr std::array<const char *,60> keys{{"packed_bgra8","current_source_bgra8","Base","state","ownerM","ownerRGB","SCP","SCT","b2_actual_upload","b3_actual_upload","withdraw_actual_upload","Alive","current_display_source","current_source_receipt","current_source_constants_actual_upload","current_cue_body","current_cue_matte","current_cue_receipt","cue_input_current_source","cue_input_current_constants","cue_input_prior_ocr","cue_input_prior_constants","frozen_A_source","frozen_A_ocr","frozen_A_constants","frozen_A_uip","frozen_A_sct","frozen_A_transaction","frozen_A_dispatch_arguments","retirement_A_source","retirement_A_P","retirement_A_ocr","retirement_A_uip","retirement_A_sct","retirement_A_constants","retirement_A_transaction","retirement_A_arguments","retirement_A_state","retirement_A_anchors","retirement_A_scratch","retirement_A_scores","retirement_A_negative","retirement_A_expected","retirement_A_watermark","retirement_B_source","retirement_B_P","retirement_B_ocr","retirement_B_uip","retirement_B_sct","retirement_B_constants","retirement_B_transaction","retirement_B_arguments","retirement_B_state","retirement_B_anchors","retirement_B_scratch","retirement_B_scores","retirement_B_negative","retirement_B_expected","retirement_B_watermark","pixel_route_control"}};
        constexpr std::array<const char *,60> names{{"packed.bgra8","current-source.bgra8","Base.f32","state.u32","ownerM.u32","ownerRGB.rgba16f","SCP.u32","SCT.u32","b2.u32","b3.u32","withdraw.u32","Alive.u32","current-display-source.rgba16f","current-source-receipt.u32","current-source-constants.u32","current-cue-body.u32","current-cue-M.u32","current-cue-receipt.u32","cue-input-current-source.rgba16f","cue-input-current-context.u32","cue-input-prior-ocr.u32","cue-input-prior-context.u32","frozen-A-source.rgba16f","frozen-A-ocr.u32","frozen-A-context.u32","frozen-A-UIP.u32","frozen-A-SCT.u32","frozen-A-transaction.u32","frozen-A-dispatch-arguments.u32","retirement-A-source.rgba16f","retirement-A-P.f32","retirement-A-ocr.u32","retirement-A-uip.u32","retirement-A-sct.u32","retirement-A-constants.u32","retirement-A-transaction.u32","retirement-A-arguments.u32","retirement-A-state.u32","retirement-A-anchors.u32","retirement-A-scratch.u32","retirement-A-scores.u32","retirement-A-negative.u32","retirement-A-expected.u32","retirement-A-watermark.u32","retirement-B-source.rgba16f","retirement-B-P.f32","retirement-B-ocr.u32","retirement-B-uip.u32","retirement-B-sct.u32","retirement-B-constants.u32","retirement-B-transaction.u32","retirement-B-arguments.u32","retirement-B-state.u32","retirement-B-anchors.u32","retirement-B-scratch.u32","retirement-B-scores.u32","retirement-B-negative.u32","retirement-B-expected.u32","retirement-B-watermark.u32","pixel-route-control.bgra8"}};
        const std::size_t active_key_count=retirement_enabled?60u:cue_enabled?29u:snapshot_enabled?15u:12u;
        std::array<jp_stage_t,60> stages;
        std::array<jp_capture_t,5> captures;
        std::vector<std::pair<std::string,nlohmann::json>> attempts;attempts.reserve(3);
        nlohmann::json draw_records=nlohmann::json::object(),trace=nlohmann::json::array(),A_hashes;
        std::vector<std::uint32_t> prior_alive;
        bool flushed=false;
        ComPtr<ID3D11Buffer> mutable_b3,mutable_withdraw;
        const auto now_us=[](){return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();};
        const auto make_cb=[&](UINT size,ComPtr<ID3D11Buffer> &out) {
          D3D11_BUFFER_DESC desc{};desc.ByteWidth=size;desc.Usage=D3D11_USAGE_DEFAULT;desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
          jp_require(dev->CreateBuffer(&desc,nullptr,&out),"Probe preallocated mutable CB failed");
        };
        const auto prepare_stage=[&](std::size_t index,ID3D11Resource *source) {
          if(!source)throw std::runtime_error("Warmup layout source missing");
          auto &s=stages.at(index);source->GetType(&s.dimension);
          if(s.dimension==D3D11_RESOURCE_DIMENSION_TEXTURE2D) {
            ComPtr<ID3D11Texture2D> texture,stage;jp_require(source->QueryInterface(IID_PPV_ARGS(&texture)),"Warmup texture interface failed");texture->GetDesc(&s.texture);
            if(s.texture.MipLevels!=1||s.texture.ArraySize!=1||s.texture.SampleDesc.Count!=1)throw std::runtime_error("Unsupported warmup texture layout");
            const std::size_t pixel_bytes=s.texture.Format==DXGI_FORMAT_R16G16B16A16_FLOAT?8u:4u;
            if(s.texture.Format!=DXGI_FORMAT_R16G16B16A16_FLOAT&&s.texture.Format!=DXGI_FORMAT_R32_FLOAT&&s.texture.Format!=DXGI_FORMAT_R32_UINT&&s.texture.Format!=DXGI_FORMAT_B8G8R8A8_UNORM)throw std::runtime_error("Unexpected warmup texture format");
            s.row_bytes=std::size_t(s.texture.Width)*pixel_bytes;s.rows=s.texture.Height;
            auto desc=s.texture;desc.Usage=D3D11_USAGE_STAGING;desc.BindFlags=desc.MiscFlags=0;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
            jp_require(dev->CreateTexture2D(&desc,nullptr,&stage),"Preallocated texture staging failed");jp_require(stage.As(&s.resource),"Texture stage resource failed");
          } else if(s.dimension==D3D11_RESOURCE_DIMENSION_BUFFER) {
            ComPtr<ID3D11Buffer> buffer,stage;jp_require(source->QueryInterface(IID_PPV_ARGS(&buffer)),"Warmup buffer interface failed");D3D11_BUFFER_DESC desc{};buffer->GetDesc(&desc);
            s.buffer_bytes=desc.ByteWidth;s.row_bytes=desc.ByteWidth;s.rows=1;
            desc.Usage=D3D11_USAGE_STAGING;desc.BindFlags=desc.MiscFlags=desc.StructureByteStride=0;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
            jp_require(dev->CreateBuffer(&desc,nullptr,&stage),"Preallocated buffer staging failed");jp_require(stage.As(&s.resource),"Buffer stage resource failed");
          } else throw std::runtime_error("Unsupported warmup resource type");
          for(auto &capture:captures)capture.bytes[index].resize(s.row_bytes*s.rows);
        };
        const auto prepare_srv=[&](std::size_t index,ID3D11ShaderResourceView *view) {
          if(!view)throw std::runtime_error("Warmup layout SRV missing");ComPtr<ID3D11Resource> resource;view->GetResource(&resource);prepare_stage(index,resource.Get());
        };
        const auto capture_resource=[&](jp_capture_t &capture,std::size_t index,ID3D11Resource *source) {
          auto &s=stages.at(index);if(!source||!s.resource||capture.copied[index])throw std::runtime_error("Invalid/duplicate preallocated readback");
          D3D11_RESOURCE_DIMENSION dimension{};source->GetType(&dimension);if(dimension!=s.dimension)throw std::runtime_error("Readback resource type changed");
          if(dimension==D3D11_RESOURCE_DIMENSION_TEXTURE2D) {
            ComPtr<ID3D11Texture2D> texture;jp_require(source->QueryInterface(IID_PPV_ARGS(&texture)),"Readback texture interface failed");D3D11_TEXTURE2D_DESC actual{};texture->GetDesc(&actual);
            if(actual.Width!=s.texture.Width||actual.Height!=s.texture.Height||actual.Format!=s.texture.Format||actual.MipLevels!=1||actual.ArraySize!=1||actual.SampleDesc.Count!=1)throw std::runtime_error("Readback texture layout changed after warmup");
          } else {
            ComPtr<ID3D11Buffer> buffer;jp_require(source->QueryInterface(IID_PPV_ARGS(&buffer)),"Readback buffer interface failed");D3D11_BUFFER_DESC actual{};buffer->GetDesc(&actual);if(actual.ByteWidth!=s.buffer_bytes)throw std::runtime_error("Readback buffer layout changed after warmup");
          }
          if(capture.bytes[index].size()!=s.row_bytes*s.rows)throw std::runtime_error("Immutable CPU capture allocation changed");
          ctx->CopyResource(s.resource.Get(),source);D3D11_MAPPED_SUBRESOURCE mapped{};
          jp_require(ctx->Map(s.resource.Get(),0,D3D11_MAP_READ,0,&mapped),"Actual GPU readback Map failed");
          const auto pitch=dimension==D3D11_RESOURCE_DIMENSION_TEXTURE2D?mapped.RowPitch:static_cast<UINT>(s.row_bytes);
          for(std::size_t y=0;y<s.rows;++y)std::memcpy(capture.bytes[index].data()+y*s.row_bytes,static_cast<const char *>(mapped.pData)+y*pitch,s.row_bytes);
          ctx->Unmap(s.resource.Get(),0);capture.copied[index]=true;
        };
        const auto capture_srv=[&](jp_capture_t &capture,std::size_t index,ID3D11ShaderResourceView *view) {
          if(!view)throw std::runtime_error("Actual readback SRV missing");ComPtr<ID3D11Resource> resource;view->GetResource(&resource);capture_resource(capture,index,resource.Get());
        };
        probe::callbacks_t callbacks;
        callbacks.production_draw_and_GPU_readback=[&](ID3D11ShaderResourceView *current,const models::estimate_result &owner,const probe::phase_record_t &row) {
          const auto name=jp_phase(row.phase);
          auto &capture=captures.at(static_cast<std::size_t>(row.phase));
          if(capture.readback_complete)throw std::runtime_error("Duplicate immutable phase capture");
          if(row.current_colour_frame<1||row.current_colour_frame>3||current!=inputs[row.current_colour_frame-1].srv.Get())throw std::runtime_error("Probe source texture identity mismatch");
          if(owner.input_region.left||owner.input_region.top||owner.input_region.width()!=width||owner.input_region.height()!=height||owner.input_region.source_width!=width||owner.input_region.source_height!=height||!owner.shadow_base_final_parallax||!owner.shadow_state||!owner.subtitle_glyph_matte||!owner.subtitle_contour_source||!owner.subtitle_condition_params||!owner.subtitle_contour_identity) throw std::runtime_error("Probe needs complete authenticated full-source owner tuple");
          const std::array<std::uint32_t,4> content{{owner.field_content.left,owner.field_content.top,owner.field_content.right,owner.field_content.bottom}};
          if(row.phase==probe::phase_e::matched_A) {domain.field_width=owner.field_width;domain.field_height=owner.field_height;domain.tensor_content_ltrb=content;domain.analysis_authority=static_cast<std::uint32_t>(owner.input_region.authority);}
          else if(domain.field_width!=static_cast<std::uint32_t>(owner.field_width)||domain.field_height!=static_cast<std::uint32_t>(owner.field_height)||domain.tensor_content_ltrb!=content) throw std::runtime_error("Probe held geometry/domain changed");
          std::optional<snapshot::constants_t> submitted_snapshot_constants;
          const auto dispatch_current_snapshot=[&](){
          if(snapshot_enabled) {
            const auto &input=packets.at(row.current_colour_frame-1);
            const auto current_desc=jp_desc(current);
            const auto stamp=std::chrono::duration_cast<std::chrono::microseconds>(row.current_colour_source_stamp.time_since_epoch()).count();
            if(!snapshot_epoch_us||!snapshot_permission_id||stamp<=0||
               row.current_colour_conversion_serial!=input.conversion_serial||row.current_colour_source_stamp!=input.authored_capture_source_stamp||
               current_desc.Format!=DXGI_FORMAT_B8G8R8A8_UNORM||current_desc.Width!=width||current_desc.Height!=height||
               owner.field_width!=snapshot_field_width||owner.field_height!=snapshot_field_height||content!=snapshot_content||
               owner.input_region.analysis_generation!=snapshot_analysis_generation||owner.publication.domain_tag!=snapshot_domain_tag)
              throw std::runtime_error("Current-source authoring permission/domain/clock changed");
            const auto split=[](std::uint64_t v){return std::array<std::uint32_t,2>{static_cast<std::uint32_t>(v),static_cast<std::uint32_t>(v>>32)};};
            snapshot::constants_t c;
            c.control={1u,snapshot::tag,1u,0u};c.source_rect={0u,0u,width,height};
            c.band_field={height-snapshot_band_height,snapshot_band_height,snapshot_field_width,snapshot_field_height};c.content=snapshot_content;
            const auto serial=split(row.current_colour_conversion_serial),clock=split(static_cast<std::uint64_t>(stamp));
            const auto epoch_words=split(snapshot_epoch_us),generation=split(snapshot_analysis_generation),permission=split(snapshot_permission_id),domain_words=split(snapshot_domain_tag);
            c.current={serial[0],serial[1],clock[0],clock[1]};c.scope={epoch_words[0],epoch_words[1],generation[0],generation[1]};
            c.domain={permission[0],permission[1],domain_words[0],domain_words[1]};
            std::string error;
            if(!current_snapshot.dispatch(ctx,{snapshot_shaders[0].Get(),snapshot_shaders[1].Get(),snapshot_shaders[2].Get()},current,c,error))
              throw std::runtime_error("Current-source transport unavailable: "+error);
            submitted_snapshot_constants=c; // CPU declaration only; actual CB is independently Mapped below.
          }
          };
          if(retirement_enabled)dispatch_current_snapshot();
          int retirement_bank=-1;std::optional<retirement::expected_t> retirement_expected;
          if(retirement_enabled) {
            if(!submitted_snapshot_constants)throw std::runtime_error("Retirement CURRENT CSP is unavailable before draw");
            retirement_expected=retirement::make_expected(*submitted_snapshot_constants,owner.publication);
            std::string error;
            if(row.phase==probe::phase_e::matched_A||row.phase==probe::phase_e::late_B_geometry_current_empty) {
              const UINT bank=row.phase==probe::phase_e::matched_A?0u:1u;
              const auto actual=est.completed_ocr_conditioner_working_views_for_evaluation(owner.publication);
              if(!actual||!actual->probability_snapshot)throw std::runtime_error("Genuine current-owner completed probability snapshot unavailable");
              if(!old_group_retirement.capture_owner(bank,ctx,*actual,owner.publication,*retirement_expected,error))throw std::runtime_error(error);
            }
            for(UINT bank=0;bank<2u;++bank)if(old_group_retirement.captured(bank)&&old_group_retirement.publication(bank)==owner.publication)retirement_bank=static_cast<int>(bank);
            if(retirement_bank>=0){const auto view=current_snapshot.working_views();if(!old_group_retirement.observe(static_cast<UINT>(retirement_bank),ctx,view.source.Get(),view.receipt.Get(),current_snapshot.working_constants_buffer().Get(),*retirement_expected,error))throw std::runtime_error(error);}
          }
          ID3D11ShaderResourceView *matte=owner.subtitle_glyph_matte.Get();ComPtr<ID3D11Buffer> withdraw_cb;
          if(row.presentation_only_receipt) {
            const auto &receipt=*row.presentation_only_receipt;const auto native=jp_desc(owner.subtitle_glyph_matte.Get());
            if(receipt.estimator_generation!=owner.publication.estimator_generation ||
               receipt.geometry_publication_token!=owner.publication.publication_sequence ||
               receipt.colour_conversion_serial!=row.current_colour_conversion_serial ||
               receipt.geometry_completed_conversion_serial!=row.geometry_completed_conversion_serial)
              throw std::runtime_error("V4 presentation receipt identity mismatch");
            if(native.Format!=DXGI_FORMAT_R32_UINT||native.Width!=width||native.Height>height) throw std::runtime_error("Probe native owner M shape invalid");
            if(!alive[0].texture)throw std::runtime_error("Alive was not preallocated before authored epoch");
            const auto allocation=jp_desc(alive[0].srv.Get());if(allocation.Width!=native.Width||allocation.Height!=native.Height)throw std::runtime_error("Probe native scratch shape changed");
            const bool initialise=receipt.alive_action==held::alive_action_e::initialise_from_new_owner_then_withdraw;
            if(initialise) {alive_index=-1;prior_alive.clear();alive_publication=owner.publication.publication_sequence;alive_generation=owner.publication.estimator_generation;}
            else if(alive_index<0||alive_publication!=owner.publication.publication_sequence||alive_generation!=owner.publication.estimator_generation) throw std::runtime_error("Probe scratch ancestry invalid");
            if(receipt.alive_action!=held::alive_action_e::hold_same_capture) {
              const int next=alive_index==0?1:0;
              const std::array<std::uint32_t,8> values{{owner.input_region.left,owner.input_region.top,height-native.Height,0,native.Width,native.Height,initialise?1u:0u,0}};
              withdraw_cb=mutable_withdraw;ctx->UpdateSubresource(withdraw_cb.Get(),0,nullptr,values.data(),0,0);
              ID3D11ShaderResourceView *reads[]{current,owner.subtitle_contour_source.Get(),owner.subtitle_glyph_matte.Get(),alive_index>=0?alive[alive_index].srv.Get():nullptr};
              ID3D11UnorderedAccessView *write=alive[next].uav.Get();ID3D11Buffer *constants=withdraw_cb.Get();
              ctx->CSSetShader(withdraw.Get(),nullptr,0);ctx->CSSetShaderResources(0,4,reads);ctx->CSSetUnorderedAccessViews(0,1,&write,nullptr);ctx->CSSetConstantBuffers(0,1,&constants);ctx->Dispatch((native.Width+15)/16,(native.Height+15)/16,1);
              ID3D11ShaderResourceView *null_reads[4]{};ID3D11UnorderedAccessView *null_uav=nullptr;ID3D11Buffer *null_cb=nullptr;
              ctx->CSSetShaderResources(0,4,null_reads);ctx->CSSetUnorderedAccessViews(0,1,&null_uav,nullptr);ctx->CSSetConstantBuffers(0,1,&null_cb);ctx->CSSetShader(nullptr,nullptr,0);alive_index=next;
            }
            matte=alive[alive_index].srv.Get();
          }
          const models::host_sbs_subtitle_render_constants_t subtitle { .enabled=!owner.subtitle_work_suppressed?1u:0u,.color_mode=0,
            .analysis_generation_low=static_cast<std::uint32_t>(owner.input_region.analysis_generation),.analysis_generation_high=static_cast<std::uint32_t>(owner.input_region.analysis_generation>>32),
            .source_left=0,.source_top=0,.source_width=width,.source_height=height};
          const auto b3=mutable_b3;ctx->UpdateSubresource(b3.Get(),0,nullptr,&subtitle,0,0);
          const platf::dxgi::host_sbs_v2_draw_command_t command{.render_targets={packed_rtv.Get(),nullptr},.render_target_count=1,.vertex_shader=vs,.pixel_shader=ps,.viewport=viewport,.sampler=sampler,
            .shader_resources={current,owner.shadow_base_final_parallax.Get(),owner.shadow_state.Get(),matte,owner.subtitle_contour_source.Get(),owner.subtitle_condition_params.Get(),owner.subtitle_contour_identity.Get()},.geometry_constants=b2.Get(),.subtitle_constants=b3.Get()};
          if(retirement_enabled) {
            auto control=command;control.render_targets={retirement_control_rtv.Get(),nullptr};control.pixel_shader=ps;
            {retirement::scoped_ps_negative_t unbound;if(!unbound.bind(ctx,nullptr,nullptr))throw std::runtime_error("Cannot scope negative-free control draw");if(!platf::dxgi::record_host_sbs_v2_draw(ctx,control))throw std::runtime_error("Negative-free production recorder rejected operands");}
            auto treatment=command;treatment.pixel_shader=retirement_ps.Get();
            {retirement::scoped_ps_negative_t gate;auto *cb=retirement_bank>=0?old_group_retirement.expected_buffer(static_cast<UINT>(retirement_bank)):nullptr;auto *packet=retirement_bank>=0?old_group_retirement.negative_receipt(static_cast<UINT>(retirement_bank)):nullptr;if(!gate.bind(ctx,cb,packet))throw std::runtime_error("Cannot scope negative-only treatment draw");if(!platf::dxgi::record_host_sbs_v2_draw(ctx,treatment))throw std::runtime_error("Negative treatment production recorder rejected operands");}
          } else {
          if(!platf::dxgi::record_host_sbs_v2_draw(ctx,command))throw std::runtime_error("Probe production draw recorder rejected operands");
          }

          if(!retirement_enabled)dispatch_current_snapshot();
          bool cue_prior_present=false;
          if(cue_enabled) {
            if(row.phase==probe::phase_e::matched_A) {
              if(row.current_colour_conversion_serial!=1u||row.current_colour_frame!=1u||owner.publication.frame_id!=packets[0].estimator_frame_id||owner.publication.observation_timestamp_us!=timestamps[0])
                throw std::runtime_error("V6 A model-input/source-conversion join unavailable");
              const auto actual_A=est.completed_ocr_conditioner_working_views_for_evaluation(owner.publication);
              if(!actual_A)throw std::runtime_error("V6 genuine completed-A conditioner unavailable");
              auto mismatch=owner.publication;++mismatch.publication_sequence;
              cue_getter_mismatch_guard=!est.completed_ocr_conditioner_working_views_for_evaluation(mismatch).has_value();
              if(!cue_getter_mismatch_guard)throw std::runtime_error("V6 conditioner getter accepted mismatched publication");
              std::string error;if(!cue_pipeline.capture_A(ctx,*actual_A,owner.publication,error))throw std::runtime_error(error);
              // A is NOT conditioned on its current target OCR. Only future B/C may use this copy.
            } else {
              cue_prior_present=true;
              if(!cue_pipeline.captured_A()||est.completed_ocr_conditioner_working_views_for_evaluation(cue_pipeline.A_publication()))
                throw std::runtime_error("V6 completed-A getter did not reject pending/unmatched publication");
            }
            const auto current_view=current_snapshot.working_views();std::string error;
            if(!cue_pipeline.dispatch(ctx,current_view.source.Get(),current_view.receipt.Get(),current_snapshot.working_constants_buffer().Get(),cue_prior_present,error))
              throw std::runtime_error("V6 source-to-cue unavailable: "+error);
          }
          // All actual Map/copy operations finish inside this callback, before the
          // driver's after-query. No hashing, disk output or phase JSON occurs here.
          capture_srv(capture,0,packed.srv.Get());capture_srv(capture,1,current);
          capture_srv(capture,2,owner.shadow_base_final_parallax.Get());capture_srv(capture,3,owner.shadow_state.Get());
          capture_srv(capture,4,owner.subtitle_glyph_matte.Get());capture_srv(capture,5,owner.subtitle_contour_source.Get());
          capture_srv(capture,6,owner.subtitle_condition_params.Get());capture_srv(capture,7,owner.subtitle_contour_identity.Get());
          capture_resource(capture,8,b2.Get());capture_resource(capture,9,b3.Get());
          if(withdraw_cb)capture_resource(capture,10,withdraw_cb.Get());
          if(row.presentation_only_receipt)capture_srv(capture,11,matte);
          if(snapshot_enabled) {
            const auto view=current_snapshot.working_views();
            capture_srv(capture,12,view.source.Get());capture_resource(capture,13,view.receipt_buffer.Get());
            capture_resource(capture,14,current_snapshot.working_constants_buffer().Get());
          }
          if(cue_enabled) {
            const auto views=cue_pipeline.export_resources(current_snapshot.working_views().source.Get());
            for(std::size_t i=0;i<views.size();++i)capture_resource(capture,15u+i,views[i].Get());
          }
          const auto native=jp_desc(owner.subtitle_glyph_matte.Get());
          capture.metadata={{"GPU_readback_completed",true},{"normal_production_recorder_used",true},{"owner_bytes_never_restamped",true},
            {"source_width",width},{"source_height",height},{"packed_width",output->sbs_width},{"packed_height",output->sbs_height},
            {"owner_native_matte_shape",{native.Height,native.Width}},{"owner_field_shape",{owner.field_height,owner.field_width}},
            {"owner_content",content},{"owner_analysis_generation",owner.input_region.analysis_generation},
            {"geometry_CB_origin","shared-full-frame-factory;exactGPUupload-readback"},
            {"readback_storage","preallocated-independent-CPU-vectors;no-estimator-alias"},
            {"hash_and_disk_deferred_until_after_release",true},{"hashes_are_not_claimed_to_run_while_busy",true}};
          if(snapshot_enabled) {
            std::array<std::uint32_t,32> declaration{};std::memcpy(declaration.data(),&*submitted_snapshot_constants,sizeof(snapshot::constants_t));
            capture.metadata["current_source_snapshot"]={{"source_transport_only",true},{"semantic_or_current_UI_or_EMPTY_authority",false},
              {"submitted_constants_words",declaration},{"current_capture_epoch_steady_us",snapshot_epoch_us},
              {"private_authoring_permission_token",snapshot_permission_id},{"permission_kind","private-authored-three-source-immutable-owner"},
              {"source_domain_tag",snapshot_domain_tag},{"analysis_generation",snapshot_analysis_generation},
              {"source_clock_kind","actual-authored-capture-steady-us;not-retained-Base-observation"},
              {"source_closure_sha256",snapshot_closure},{"source_header_sha256",snapshot_plan.at("header").at("sha256")},
              {"actual_GPU_source_receipt_CB_readback",true},{"new_source_is_not_bound_to_renderer",true}};
          }
          if(cue_enabled)capture.metadata["native_current_rgb_cue"]={{"semantic_status","UNKNOWN"},{"no_current_UI_positive_or_EMPTY_authority",true},
            {"prior_present",cue_prior_present},{"prior_policy","genuine-completed-A-only;matched-A-canonical-no-prior"},
            {"frozen_A_publication",jp_publication(cue_pipeline.A_publication())},{"A_input_submission_conversion",1u},{"A_input_submission_model_frame",2u},{"A_input_relative_observation_us",33334u},
            {"getter_mismatched_publication_rejected",cue_getter_mismatch_guard},{"getter_pending_or_later_A_query_rejected",row.phase!=probe::phase_e::matched_A},
            {"pipeline_header",cue_plan.at("pipeline_header")},{"cue_source_closure_sha256",cue_plan.at("cue_source_closure_sha256")},
            {"component_source_closure_sha256",cue_plan.at("component_source_closure_sha256")},{"context_wire_source_closure_sha256",cue_plan.at("context_wire_source_closure_sha256")},
            {"current_context_origin","GPU-CSP128+GPU-sourceReceipt+geometry-only-template;not-CPU-model-clock"},
            {"new_cue_is_not_bound_to_renderer",true},{"copied_GPU_A_context_not_AddRef_snapshot",true}};
          if(retirement_enabled) {
            for(UINT bank=0;bank<2u;++bank){const auto operands=old_group_retirement.export_resources(bank);for(std::size_t i=0;i<operands.size();++i)capture_resource(capture,29u+bank*15u+i,operands[i].Get());}
            capture_srv(capture,59,retirement_control.srv.Get());
            capture.metadata["source_pattern_retirement"]={{"scope","NEGATIVE_OLD_GROUP_ONLY"},{"semantic_presence","UNKNOWN"},{"no_current_EMPTY_or_positive",true},{"owner_bank",retirement_bank},{"A_captured",old_group_retirement.captured(0)},{"B_captured",old_group_retirement.captured(1)},
             {"rule_freeze_sha256",retirement_plan.at("rule_freeze_sha256")},{"GPU_source_closure",retirement_closure},{"private_PS_closure",retirement_render_closure},{"control_and_treatment_same_current_RGB_owner_M_alive",true},{"normal_recorder_unchanged",true},{"current_watermark_not_reset_on_B",true},{"all_Map_before_after600",true}};
          }
          capture.readback_complete_wall_us=now_us();capture.readback_complete=true;
        };
        callbacks.append_attempt_diagnostic=[&](const probe::phase_record_t &row,const held::geometry_owner_t &owner,
          const held::captured_colour_t &colour,const held::current_evidence_t &evidence) {
          const auto stamp=[](auto t){return std::chrono::duration_cast<std::chrono::microseconds>(t.time_since_epoch()).count();};
          const auto domain_json=[](const held::domain_t &d){return nlohmann::json{
            {"source",{d.source_width,d.source_height}},{"format",d.format},{"mips",d.mip_levels},{"array_size",d.array_size},
            {"sample_count",d.sample_count},{"sample_quality",d.sample_quality},{"transfer",d.transfer},
            {"analysis_ltrb",d.analysis_ltrb},{"tensor_content_ltrb",d.tensor_content_ltrb},{"field",{d.field_width,d.field_height}},
            {"analysis_authority",d.analysis_authority},{"root_authority_generation",d.root_authority_generation},
            {"region_authority_generation",d.region_authority_generation},{"browser_authority_epoch",d.browser_authority_epoch},
            {"subtitle_context_generation",d.subtitle_context_generation}};};
          const auto &choice=*row.presentation_only_receipt;
          const char *route=choice.route==held::route_e::presentation_only_held_geometry_current_color?
            "presentation_only_held_geometry_current_color":choice.route==held::route_e::current_flat?"current_flat":
            choice.route==held::route_e::reject_stale_colour?"reject_stale_colour":"ordinary_existing_policy";
          attempts.emplace_back("attempt_"+jp_phase(row.phase)+".json",nlohmann::json{{"schema",3},{"phase",jp_phase(row.phase)},
            {"source_conversion_id",colour.conversion_serial},{"owner_estimator_frame_id",owner.completed_colour_frame},
            {"typed_policy_version",2},{"owner_estimator_generation",owner.estimator_generation},{"owner_publication_token",owner.publication_token},
            {"owner_completed_conversion_serial",owner.completed_colour_conversion_serial},
            {"selected_colour_conversion_serial",choice.colour_conversion_serial},
            {"selected_colour_source_stamp_us",stamp(choice.colour_source_stamp)},
            {"observed_at_wall_us",stamp(colour.observed_at)},{"accepted_owner_wall_us",stamp(owner.accepted_capture_at)},
            {"owner_deadline_wall_us",stamp(owner.accepted_capture_at+held::max_owner_age)},
            {"owner_age_us",std::chrono::duration_cast<std::chrono::microseconds>(colour.observed_at-owner.accepted_capture_at).count()},
            {"current_source_stamp_us",stamp(colour.capture_source_stamp)},{"owner_source_stamp_us",stamp(owner.completed_colour_source_stamp)},
            {"domain_equal",colour.domain==owner.domain},{"owner_domain",domain_json(owner.domain)},{"current_domain",domain_json(colour.domain)},
            {"source_stamp_present",colour.source_stamp_present},{"texture_owned",colour.current_texture_owned_until_draw_submission},
            {"joined_root_busy",evidence.joined_root_busy},{"publication_still_current",evidence.publication_still_current},
            {"structural_join",evidence.structural_route_join_verified},{"producer_terminal",evidence.producer_terminal},
            {"move_size_or_reprocess",evidence.move_size_or_reprocess},{"current_cut",static_cast<unsigned>(evidence.current_cut)},
            {"selected_route",route},{"gate_before",jp_gate(row.gate_before)},{"diagnostic_only_no_new_authority",true}});
        };
        const auto publish_trace=[&](const probe::phase_record_t &row) {
          const auto name=jp_phase(row.phase);auto j=draw_records.at(name);j["phase"]=name;j["current_colour_frame"]=row.current_colour_frame;j["current_source_sha256"]=row.source_file_sha256;
          j["typed_policy_version"]=2;j["typed_policy_source_sha256"]="7eb1e2b8c63ce042f9426788a99becd8dbd50cc949d306e9363ebd0dead5c2cf";
          j["current_colour_conversion_serial"]=row.current_colour_conversion_serial;
          j["geometry_completed_conversion_serial"]=row.geometry_completed_conversion_serial;
          j["current_authored_capture_stamp_us"]=std::chrono::duration_cast<std::chrono::microseconds>(row.current_colour_source_stamp.time_since_epoch()).count();
          j["tuple_completed_colour_frame"]=row.completed_geometry_frame;j["actual_depth_owner_frame"]=row.actual_depth_owner_frame;j["actual_Base_clock_us"]=row.actual_Base_clock_us?nlohmann::json(*row.actual_Base_clock_us):nlohmann::json(nullptr);
          j["geometry_publication"]=jp_publication(row.geometry_publication);j["gate_before"]=jp_gate(row.gate_before);j["gate_after"]=jp_gate(row.gate_after);j["before_query_return_wall_us"]=row.before_query_return_wall_us;j["after_query_return_wall_us"]=row.after_query_return_wall_us;
          if(row.decoded_owner_receipt_after_draw) {
            const auto &r=*row.decoded_owner_receipt_after_draw;
            j["real_decoded_owner_receipt_after_draw"]={{"expected",jp_publication(r.expected)},{"geometry_valid",r.geometry_valid},{"depth_owner_valid",r.depth_owner_valid},
              {"depth_owner_frame_id",r.depth_owner_frame_id},{"depth_owner_timestamp_us",r.depth_owner_timestamp_us},{"depth_disposition",static_cast<unsigned>(r.depth)},
              {"subtitle_disposition",static_cast<unsigned>(r.subtitle)},{"subtitle_frame_id",r.subtitle_frame_id},{"subtitle_domain_tag",r.subtitle_domain_tag}};
          }
          j["current_cut_verdict"]=row.presentation_only_receipt?"UNKNOWN":"normal-joined-completion";j["exact_depth_reuse"]=false;j["new_subtitle_observation_from_held_draw"]=false;j["authored_capture_schedule_not_live_DDup"]=true;
          if(row.presentation_only_receipt) {const auto &v=*row.presentation_only_receipt;j["presentation_only_receipt"]={{"route","presentation_only_held_geometry_current_color"},{"geometry_publication_token",v.geometry_publication_token},{"estimator_generation",v.estimator_generation},{"colour_conversion_serial",v.colour_conversion_serial},{"geometry_completed_conversion_serial",v.geometry_completed_conversion_serial},{"tuple_completed_colour_frame",v.tuple_completed_colour_frame},{"withdrawal_revision",v.withdrawal_revision},{"alive_action",static_cast<unsigned>(v.alive_action)},{"owner_deadline_wall_us",std::chrono::duration_cast<std::chrono::microseconds>(v.owner_deadline.time_since_epoch()).count()}};}
          jp_json(root/name/"phase.json",j);trace.push_back({{"phase",name},{"metadata",jp_file(root/name/"phase.json")}});
        };
        callbacks.append_immutable_trace=[&](const probe::phase_record_t &row) {
          auto &capture=captures.at(static_cast<std::size_t>(row.phase));
          if(!capture.readback_complete||capture.authenticated_row)throw std::runtime_error("Missing/duplicate immutable GPU capture");
          if(row.gate_after&&capture.readback_complete_wall_us>row.after_query_return_wall_us)throw std::runtime_error("Readback did not precede actual after-query");
          capture.authenticated_row=row; // Immutable CPU receipt copy; no owner GPU read here.
        };
        flush_deferred=[&]() {
          if(flushed)return;
          flushed=true;
          const auto at_flush=est.joined_completion_gate_status_for_benchmark();
          if(at_flush&&at_flush->callback_entered&&!at_flush->released)throw std::runtime_error("Diagnostic flush attempted while gate held");
          for(const auto &attempt:attempts)jp_json(root/attempt.first,attempt.second);
          // Save EVERY completed raw Map copy before any semantic comparison can throw.
          for(std::size_t phase=0;phase<captures.size();++phase) {
            auto &capture=captures[phase];bool any=false;for(bool copied:capture.copied)any|=copied;if(!any)continue;
            const auto name=jp_phase(static_cast<probe::phase_e>(phase));const auto folder=root/name;
            if(!fs::create_directory(folder))throw std::runtime_error("Deferred phase folder already exists");
            nlohmann::json files;
            for(std::size_t k=0;k<keys.size();++k)if(capture.copied[k]) {
              const auto &bytes=capture.bytes[k];const auto file=folder/names[k];
              std::ofstream out(file,std::ios::binary);out.write(bytes.data(),static_cast<std::streamsize>(bytes.size()));out.flush();if(!out.good())throw std::runtime_error("Deferred immutable byte export failed");out.close();
              files[keys[k]]={{"path",fs::absolute(file).lexically_normal().string()},{"bytes",bytes.size()},
                {"sha256",sha256_hex(std::string_view(bytes.data(),bytes.size()))}};
            }
            auto metadata=capture.metadata;metadata["artifacts"]=files;metadata["GPU_readback_complete_wall_us"]=capture.readback_complete_wall_us;
            metadata["diagnostic_hash_flush_wall_us"]=now_us();metadata["post_readback_query_verified"]=capture.authenticated_row.has_value();
            draw_records[name]=metadata;
            if(!capture.authenticated_row)jp_json(folder/"unpublished-partial.json",metadata);
          }
          for(std::size_t phase=0;phase<captures.size();++phase) {
            auto &capture=captures[phase];if(!capture.authenticated_row)continue;const auto &row=*capture.authenticated_row;
            const auto name=jp_phase(row.phase);auto &metadata=draw_records[name];const auto &files=metadata["artifacts"];
            if(files["current_source_bgra8"]["sha256"].get<std::string>()!=upload_hashes[row.current_colour_frame-1])throw std::runtime_error("Deferred current source pixels differ from immutable upload");
            if(snapshot_enabled) {
              if(!capture.copied[12]||!capture.copied[13]||!capture.copied[14]||capture.bytes[13].size()!=256u||capture.bytes[14].size()!=128u)
                throw std::runtime_error("Missing immutable current-source readbacks");
              std::array<std::uint32_t,64> receipt{};std::array<std::uint32_t,32> actual_cb{};
              std::memcpy(receipt.data(),capture.bytes[13].data(),256u);std::memcpy(actual_cb.data(),capture.bytes[14].data(),128u);
              const auto declared=capture.metadata.at("current_source_snapshot").at("submitted_constants_words").get<std::array<std::uint32_t,32>>();
              const auto actual_clock=std::uint64_t{actual_cb[18]}|(std::uint64_t{actual_cb[19]}<<32);
              const auto actual_serial=std::uint64_t{actual_cb[16]}|(std::uint64_t{actual_cb[17]}<<32);
              const auto row_clock=std::chrono::duration_cast<std::chrono::microseconds>(row.current_colour_source_stamp.time_since_epoch()).count();
              if(actual_cb!=declared||!std::equal(actual_cb.begin(),actual_cb.end(),receipt.begin()+8)||
                 receipt[0]!=snapshot::tag||receipt[1]!=1u||receipt[2]!=1u||receipt[3]!=0u||receipt[4]!=width||receipt[5]!=snapshot_band_height||
                 receipt[6]!=width*snapshot_band_height||receipt[7]!=0u||receipt[40]!=width||receipt[41]!=height||receipt[42]!=width||receipt[43]!=snapshot_band_height||
                 receipt[44]!=0u||receipt[45]!=1u||std::any_of(receipt.begin()+46,receipt.end(),[](auto v){return v!=0u;})||
                 actual_serial!=row.current_colour_conversion_serial||actual_clock!=static_cast<std::uint64_t>(row_clock))
                throw std::runtime_error("Current-source actual receipt/CB/current-clock validation failed");
              metadata["current_source_snapshot"]["actual_receipt_ready"]=true;
              metadata["current_source_snapshot"]["actual_CB_and_receipt_mirrors_exact"]=true;
            }
            if(cue_enabled) {
              for(std::size_t k=15;k<29;++k)if(!capture.copied[k])throw std::runtime_error("Missing V6 source-to-cue immutable readback");
              const auto words=[&](std::size_t k,std::size_t n){if(capture.bytes[k].size()!=n*4u)throw std::runtime_error("V6 typed extent mismatch");std::vector<std::uint32_t> v(n);std::memcpy(v.data(),capture.bytes[k].data(),n*4u);return v;};
              const auto current_cb=words(19,24),prior_ocr=words(20,256),prior_cb=words(21,24),packet=words(17,512),source_cb=words(14,32);
              const auto Aocr=words(23,256),Acb=words(24,24),Auip=words(25,40),Asct=words(26,32),At=words(27,64),Aa=words(28,64);
              const auto clock=[](const auto &v,std::size_t k){return std::uint64_t{v[k]}|(std::uint64_t{v[k+1]}<<32);};
              if(files["cue_input_current_source"]["sha256"]!=files["current_display_source"]["sha256"]||
                 !std::equal(current_cb.begin(),current_cb.begin()+12,Acb.begin())||current_cb[12]!=source_cb[16]||current_cb[13]!=source_cb[17]||current_cb[14]!=source_cb[22]||current_cb[15]!=source_cb[23]||
                 current_cb[16]!=source_cb[18]||current_cb[17]!=source_cb[19]||current_cb[18]!=0u||current_cb[19]!=0u||current_cb[20]!=1u||current_cb[21]!=0u||current_cb[22]!=0u||current_cb[23]!=0u||
                 packet[0]!=0x31475243u||packet[1]!=1u||packet[2]!=0u||!std::equal(current_cb.begin(),current_cb.end(),packet.begin()+16)||!std::equal(prior_cb.begin(),prior_cb.end(),packet.begin()+40))
                throw std::runtime_error("V6 actual CSP/currentCB/cue-receipt input join failed");
              if(Aocr[0]!=4u||Aocr[1]!=0x3852434fu||Aocr[2]!=1u||Aocr[3]!=Aocr[4]||Aocr[4]==0u||Aocr[4]>8u||
                 clock(Acb,12)!=2u||clock(Acb,16)!=33334u||!std::equal(Aocr.begin()+5,Aocr.begin()+9,Acb.begin()+12)||
                 Asct[0]!=5u||Asct[1]!=0x35544353u||Asct[23]!=1u||Asct[22]!=27u||!std::equal(Asct.begin()+4,Asct.begin()+8,Acb.begin()+12)||
                 !std::equal(Asct.begin()+8,Asct.begin()+20,Acb.begin())||!std::equal(Asct.begin()+20,Asct.begin()+22,Acb.begin()+16)||
                 !std::equal(Auip.begin()+4,Auip.begin()+10,Acb.begin()+12)||Auip[20]!=Asct[3]||
                 At!=Aa||At[0]!=1u||At[7]!=0x52434f4fu||At[13]!=1u||At[1]!=(At[0]^0xd1ec15a5u^At[7])||At[6]!=0x47524243u||At[12]!=0x54535152u||
                 clock(At,2)!=cue_pipeline.A_publication().transaction_token||At[2]!=At[8]||At[3]!=At[9]||At[4]!=(At[2]^0xa3756c91u)||At[5]!=(At[3]^0x5c8a936eu)||At[10]!=At[4]||At[11]!=At[5]||At[14]!=(1u^0x6f435257u)||At[15]!=0u||
                 At[60]!=1u||At[61]!=1u||At[62]!=1u||At[52]!=1u||At[53]!=1u||At[54]!=1u||At[56]!=0u||At[57]!=1u||At[58]!=1u)
                throw std::runtime_error("V6 genuine A own OCR/CB/UIP/SCT/transaction join failed");
              nlohmann::json Ahash;for(const auto *k:{"frozen_A_source","frozen_A_ocr","frozen_A_constants","frozen_A_uip","frozen_A_sct","frozen_A_transaction","frozen_A_dispatch_arguments"})Ahash[k]=files[k]["sha256"];
              const bool prior=row.phase!=probe::phase_e::matched_A;
              if(!prior) {
                frozen_A_hashes=Ahash;
                if(files["frozen_A_source"]["sha256"]!=files["ownerRGB"]["sha256"]||files["frozen_A_source"]["sha256"]!=files["current_display_source"]["sha256"]||files["frozen_A_sct"]["sha256"]!=files["SCT"]["sha256"]||
                   std::any_of(prior_cb.begin(),prior_cb.end(),[](auto v){return v!=0u;})||std::any_of(prior_ocr.begin(),prior_ocr.end(),[](auto v){return v!=0u;}))throw std::runtime_error("V6 A must freeze genuine source while cue uses canonicalzero prior");
              } else if(Ahash!=frozen_A_hashes||prior_cb!=Acb||prior_ocr!=Aocr)throw std::runtime_error("V6 immutable A changed or current cue used another prior");
              const auto body=words(15,std::size_t(width)*snapshot_band_height),matte=words(16,std::size_t(width)*snapshot_band_height);
              std::uint32_t bc=0,mc=0;for(std::size_t p=0;p<body.size();++p){if(body[p]>1u||matte[p]>1u||body[p]>matte[p])throw std::runtime_error("V6 cue output is not binary/subset");bc+=body[p];mc+=matte[p];}
              if(packet[8]!=bc||packet[9]!=mc||packet[4]!=1u||(packet[3]&3u)!=0u||(packet[3]!=0u&&(bc!=0u||mc!=0u)))throw std::runtime_error("V6 cue typed/numeric/fault/count validation failed");
              metadata["native_current_rgb_cue"]["actual_body_count"]=bc;metadata["native_current_rgb_cue"]["actual_M_count"]=mc;
              metadata["native_current_rgb_cue"]["actual_fault_bits"]=packet[3];metadata["native_current_rgb_cue"]["algorithm_context_usable"]=packet[3]==0u;
              metadata["native_current_rgb_cue"]["frozen_A_hashes"]=Ahash;
            }
            nlohmann::json hashes;for(const auto *key:{"Base","state","ownerM","ownerRGB","SCP","SCT"})hashes[key]=files[key]["sha256"];
            metadata["owner_tuple_hashes"]=hashes;
            const auto n=capture.bytes[4].size()/4;std::vector<std::uint32_t> original(n);std::memcpy(original.data(),capture.bytes[4].data(),n*4);
            if(row.phase==probe::phase_e::matched_A) {
              A_hashes=hashes;if(std::none_of(original.begin(),original.end(),[](auto value){return value==1;}))throw std::runtime_error("A has no actual old glyph protection to revoke");
            }
            if((row.phase==probe::phase_e::busy_B_current_colour||row.phase==probe::phase_e::busy_empty_current_colour)&&hashes!=A_hashes)throw std::runtime_error("Immutable busy owner bytes differ from A");
            if(capture.copied[11]) {
              if(capture.bytes[11].size()!=n*4)throw std::runtime_error("Deferred Alive extent mismatch");
              std::vector<std::uint32_t> current(n);std::memcpy(current.data(),capture.bytes[11].data(),n*4);
              if(row.presentation_only_receipt->alive_action==held::alive_action_e::initialise_from_new_owner_then_withdraw)prior_alive.clear();
              for(std::size_t p=0;p<n;++p)if(current[p]>1||(current[p]&&original[p]!=1)||(!prior_alive.empty()&&current[p]>prior_alive[p]))throw std::runtime_error("Immutable Alive binary/subset/monotonicity failed");
              metadata["Alive_active_count"]=std::count(current.begin(),current.end(),1u);
              metadata["ownerM_active_count"]=std::count(original.begin(),original.end(),1u);
              prior_alive=std::move(current);
            }
            publish_trace(row);
          }
        };
        probe::options_t options;options.explicitly_enable_bounded_approximate_research=true;options.gate_hold_after_entry_ms=plan.at("hold_after_entry_ms").get<std::uint32_t>();options.empty_capture_after_entry_ms=plan.at("empty_after_entry_ms").get<std::uint32_t>();
        // Configure targetB3 before firstinput, genuinely infer empty1/obs1, then
        // serialize the authenticated warmup proof BEFORE establishing authored epoch.
        const auto prepared=probe::prepare(est,inputs[2].srv.Get(),options);
        // Genuine empty warmup supplies authoritative actual layout. Allocate staging,
        // Alive, mutable CBs and all independent CPU phase buffers BEFORE authored epoch.
        make_cb(32u,mutable_b3);make_cb(32u,mutable_withdraw);
        const auto warm_native=jp_desc(prepared.actual_warmup.subtitle_glyph_matte.Get());
        if(warm_native.Format!=DXGI_FORMAT_R32_UINT||warm_native.Width!=width||!warm_native.Height||warm_native.Height>height)throw std::runtime_error("Warmup native layout invalid");
        for(auto &v:alive)v=jp_texture(dev,warm_native.Width,warm_native.Height,DXGI_FORMAT_R32_UINT,D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS);
        prepare_srv(0,packed.srv.Get());prepare_srv(1,inputs[0].srv.Get());
        prepare_srv(2,prepared.actual_warmup.shadow_base_final_parallax.Get());prepare_srv(3,prepared.actual_warmup.shadow_state.Get());
        prepare_srv(4,prepared.actual_warmup.subtitle_glyph_matte.Get());prepare_srv(5,prepared.actual_warmup.subtitle_contour_source.Get());
        prepare_srv(6,prepared.actual_warmup.subtitle_condition_params.Get());prepare_srv(7,prepared.actual_warmup.subtitle_contour_identity.Get());
        prepare_stage(8,b2.Get());prepare_stage(9,mutable_b3.Get());prepare_stage(10,mutable_withdraw.Get());prepare_srv(11,alive[0].srv.Get());
        if(snapshot_enabled) {
          const auto &warm=prepared.actual_warmup;
          if(warm.input_region.left||warm.input_region.top||warm.input_region.width()!=width||warm.input_region.height()!=height||
             warm.input_region.source_width!=width||warm.input_region.source_height!=height||!warm.publication.domain_tag)
            throw std::runtime_error("Current-source warmup structural layout unavailable");
          snapshot_band_height=warm_native.Height;snapshot_field_width=warm.field_width;snapshot_field_height=warm.field_height;
          snapshot_content={warm.field_content.left,warm.field_content.top,warm.field_content.right,warm.field_content.bottom};
          snapshot_analysis_generation=warm.input_region.analysis_generation;snapshot_domain_tag=warm.publication.domain_tag;
          std::string error;if(!current_snapshot.ensure(dev,width,snapshot_band_height,error))throw std::runtime_error("Preallocated current-source unavailable: "+error);
          const auto view=current_snapshot.working_views();prepare_srv(12,view.source.Get());prepare_stage(13,view.receipt_buffer.Get());
          prepare_stage(14,current_snapshot.working_constants_buffer().Get());
        }
        if(cue_enabled) {
          const auto warm=est.completed_ocr_conditioner_working_views_for_evaluation(prepared.actual_warmup.publication);
          if(!warm)throw std::runtime_error("V6 genuine warmup conditioner layout unavailable");
          std::array<ID3D11ComputeShader*,20> c{};std::array<ID3D11ComputeShader*,3> g{};
          for(std::size_t i=0;i<c.size();++i)c[i]=cue_shaders[i].Get();for(std::size_t i=0;i<g.size();++i)g[i]=cue_graph_shaders[i].Get();
          std::string error;if(!cue_pipeline.prepare(dev,ctx,*warm,c,g,cue_context_wire.Get(),error))throw std::runtime_error("V6 pre-epoch preparation unavailable: "+error);
          const auto views=cue_pipeline.export_resources(current_snapshot.working_views().source.Get());for(std::size_t i=0;i<views.size();++i)prepare_stage(15u+i,views[i].Get());
        }
        if(retirement_enabled) {
          const auto warm=est.completed_ocr_conditioner_working_views_for_evaluation(prepared.actual_warmup.publication);if(!warm||!warm->probability_snapshot)throw std::runtime_error("Retirement warmup oldP layout unavailable");
          retirement::shader_set_t set;for(std::size_t i=0;i<set.shaders.size();++i)set.shaders[i]=retirement_shaders[i].Get();std::string error;if(!old_group_retirement.prepare(dev,ctx,*warm,set,error))throw std::runtime_error(error);
          for(UINT bank=0;bank<2u;++bank){const auto operands=old_group_retirement.export_resources(bank);for(std::size_t i=0;i<operands.size();++i)prepare_stage(29u+bank*15u+i,operands[i].Get());}
          prepare_srv(59,retirement_control.srv.Get());
        }
        nlohmann::json layouts=nlohmann::json::array();std::size_t total_cpu_bytes=0;
        for(std::size_t k=0;k<active_key_count;++k){layouts.push_back({{"key",keys[k]},{"row_bytes",stages[k].row_bytes},{"rows",stages[k].rows},{"bytes_per_phase",stages[k].row_bytes*stages[k].rows}});total_cpu_bytes+=5*stages[k].row_bytes*stages[k].rows;}
        jp_json(root/"preallocation.json",{{"schema",3},{"genuine_warmup_publication",jp_publication(prepared.actual_warmup.publication)},
          {"layouts",layouts},{"CPU_bytes_reserved",total_cpu_bytes},{"staging_and_Alive_allocated_before_epoch",true},
          {"readbacks_must_complete_before_after_query",true},{"hash_disk_phase_JSON_deferred_after_release",true}});
        const auto &wr=prepared.actual_warmup_receipt;
        jp_json(root/"prewarm.json",{{"schema",2},{"warmup_source",input_bindings.at(2)},
          {"estimator_frame_id",1},{"observation_timestamp_us",1},{"source_conversion_id",nullptr},
          {"actual_publication",jp_publication(prepared.actual_warmup.publication)},
          {"actual_receipt_expected",jp_publication(wr.expected)},{"depth_owner_frame_id",wr.depth_owner_frame_id},
          {"depth_owner_timestamp_us",wr.depth_owner_timestamp_us},{"geometry_valid",wr.geometry_valid},
          {"depth_owner_valid",wr.depth_owner_valid},{"depth_disposition",static_cast<unsigned>(wr.depth)},
          {"subtitle_disposition",static_cast<unsigned>(wr.subtitle)},
          {"actual_gate_after_prepare",jp_gate(prepared.actual_gate_after_prepare)},
          {"real_prepare_completed_wall_us",std::chrono::duration_cast<std::chrono::microseconds>(prepared.real_prepare_completed_at.time_since_epoch()).count()},
          {"authored_epoch_not_established_yet",true},{"no_A_acceptance_clock_reset",true}});
        if(snapshot_enabled) {
          // A host-created permission for these three validated immutable input SRVs, not a text/cut verdict.
          const auto permission_stamp=now_us();if(permission_stamp<=0)throw std::runtime_error("Private authoring permission stamp unavailable");
          snapshot_permission_id=static_cast<std::uint64_t>(permission_stamp);
        }
        const auto epoch=std::chrono::steady_clock::now();
        if(snapshot_enabled) {
          const auto stamp=std::chrono::duration_cast<std::chrono::microseconds>(epoch.time_since_epoch()).count();
          if(stamp<=0||static_cast<std::uint64_t>(stamp)<snapshot_permission_id)throw std::runtime_error("Invalid private authored capture epoch");
          snapshot_epoch_us=static_cast<std::uint64_t>(stamp);
        }
        for(std::size_t i=0;i<3;++i)packets[i].authored_capture_source_stamp=epoch+std::chrono::microseconds {timestamps[i]};
        nlohmann::json pre_input={{"schema",3},{"probe_branch_version",4},{"typed_policy_version",2},{"typed_policy_source_sha256","7eb1e2b8c63ce042f9426788a99becd8dbd50cc949d306e9363ebd0dead5c2cf"},{"plan_sha256",sha256_hex(plan_bytes)},{"prewarm_proof",jp_file(root/"prewarm.json")},{"estimator_frame_ids",{2,3,4}},{"source_conversion_ids",{1,2,3}},{"plan_path",fs::absolute(o.joined_completion_probe).string()},{"sources",input_bindings},{"observation_timeline",jp_file(o.observation_timeline)},{"private_withdrawal_closure",closure},{"colour_format","B8G8R8A8_UNORM SDR"},{"capture_schedule","immutable-authored-timeline;not-live-DDup"},{"UNKNOWN_current_cut",true},{"approximate_geometry_research",true},{"performance_qualified",false}};
        if(snapshot_enabled) {pre_input["probe_branch_version"]=5;pre_input["current_source_snapshot"]={{"enabled",true},{"source_transport_only",true},{"source_closure_sha256",snapshot_closure},{"header",snapshot_plan.at("header")},{"actual_capture_epoch_steady_us",snapshot_epoch_us},{"private_authoring_permission_token",snapshot_permission_id},{"semantic_or_current_UI_or_EMPTY_authority",false}};}
        if(cue_enabled) {pre_input["probe_branch_version"]=6;pre_input["native_current_rgb_cue"]={{"enabled",true},{"semantic_status","UNKNOWN"},{"rendering_authority",false},{"plan",cue_plan}};}
        jp_json(root/"pre-input.json",pre_input);
        try {
          probe::run(est,packets,domain,options,callbacks,prepared);
          flush_deferred();
        } catch(...) {
          // Captured CPU snapshots still live in this outer try scope. Release and
          // export here BEFORE unwinding their arrays; outer catch never calls the lambda.
          est.release_joined_completion_gate_for_benchmark();
          try {flush_deferred();}catch(const std::exception &copy_error){deferred_error=copy_error.what();}
          throw;
        }
        nlohmann::json completion={{"schema",3},{"probe_branch_version",4},{"typed_policy_version",2},{"typed_policy_source_sha256","7eb1e2b8c63ce042f9426788a99becd8dbd50cc949d306e9363ebd0dead5c2cf"},{"functional_driver_completed",true},{"phases",trace},{"actual_gate",jp_gate(est.joined_completion_gate_status_for_benchmark())},{"performance_qualified",false},{"normal_loop_used",false},{"live_converter_integration_tested",false},{"world_distortion_or_universal_glyph_accuracy_claim",false}};
        if(snapshot_enabled) {completion["probe_branch_version"]=5;completion["current_source_snapshot"]={{"enabled",true},{"source_transport_only",true},{"source_closure_sha256",snapshot_closure},{"header",snapshot_plan.at("header")},{"actual_capture_epoch_steady_us",snapshot_epoch_us},{"private_authoring_permission_token",snapshot_permission_id},{"semantic_or_current_UI_or_EMPTY_authority",false}};}
        if(cue_enabled) {completion["probe_branch_version"]=6;completion["native_current_rgb_cue"]={{"enabled",true},{"semantic_status","UNKNOWN"},{"rendering_authority",false},{"plan",cue_plan}};}
        jp_json(root/"completion.json",completion);
        return 0;
      } catch(const std::exception &e) {
        est.release_joined_completion_gate_for_benchmark();
        if(created&&!fs::exists(root/"failure.json")) jp_json(root/"failure.json",{{"schema",3},{"functional_driver_completed",false},{"error",e.what()},{"deferred_export_error",deferred_error},{"actual_gate",jp_gate(est.joined_completion_gate_status_for_benchmark())},{"classification","setup-or-trajectory-failure;never-a-valid-busy-proof"}});
        BOOST_LOG(error)<<"sbs-bench private joined probe: "<<e.what();return 12;
      }
    }
