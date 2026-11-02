// Included in the same private namespace after the existing jp_* helpers.
    nlohmann::json pu_gate(const models::joined_completion_gate_benchmark::snapshot_t &s) {
      const auto unmap=[](const models::joined_completion_gate_benchmark::unmap_probe_t &u) {
        return nlohmann::json{{"issued",u.issued},{"result",u.result},{"begin_wall_us",u.begin_wall_us},
          {"return_wall_us",u.return_wall_us},{"host_duration_us",u.return_wall_us-u.begin_wall_us},
          {"callback_entered_at_return_observation",u.callback_entered_when_return_observed},
          {"callback_released_at_return_observation",u.callback_released_when_return_observed}};
      };
      auto j=jp_gate(std::optional<models::joined_completion_gate_benchmark::snapshot_t>{s});
      j["placement"]=static_cast<std::uint32_t>(s.configuration.placement);
      j["hold_after_entry_ms"]=s.configuration.hold_after_callback_enter_ms;
      j["safety_timeout_ms"]=s.configuration.callback_safety_timeout_ms;
      j["optional_ocr_unmap"]=unmap(s.optional_ocr_unmap);j["depth_unmap"]=unmap(s.depth_unmap);
      j["post_unmap_query_recorded"]=s.post_unmap_query_recorded;
      j["post_unmap_query_result"]=s.post_unmap_query_result;
      j["post_unmap_query_wall_us"]=s.post_unmap_query_wall_us;return j;
    }
    int run_pre_unmap_feasibility(const opts &o,const std::vector<fs::path> &frames,
      const std::vector<std::uint64_t> &timestamps,ID3D11Device *dev,ID3D11DeviceContext *ctx,
      models::video_depth_estimator &est,const std::string &plan_bytes) {
      namespace probe=external_pre_unmap_feasibility_v1;
      const auto root=fs::path(o.out)/"pre-unmap-feasibility-v1";bool created=false;
      try {
        const auto p=nlohmann::json::parse(plan_bytes);
        if(p.at("schema")!=1||p.at("probe_kind")!="actual-pre-unmap-feasibility-v1"||
           frames.size()!=3||p.at("sources").size()!=3||timestamps!=std::vector<std::uint64_t>{33334,66667,100000}||
           p.at("observation_timestamps_us")!=nlohmann::json(timestamps)||p.at("hold_after_entry_ms")!=120||
           p.at("callback_safety_timeout_ms")!=500)throw std::runtime_error("Invalid pre-unmap measurement protocol");
        const auto placement=p.at("gate_placement").get<std::string>();
        if(placement!="before-interop-unmaps"&&placement!="after-interop-unmaps-control")
          throw std::runtime_error("Unknown gate placement");
        if(fs::exists(root))throw std::runtime_error("Pre-unmap probe refuses to overwrite output");
        fs::create_directories(root);created=true;
        std::array<jp_texture_t,3> inputs;std::array<ID3D11ShaderResourceView *,3> views{};
        std::array<std::string,3> upload_hashes;UINT width=0,height=0;
        nlohmann::json bindings=nlohmann::json::array();
        constexpr std::array<const char *,3> roles{{"A","B","empty"}};
        for(std::size_t i=0;i<3;++i) {
          const auto &b=p.at("sources").at(i);std::string bytes;rgba_image image;
          if(b.at("role")!=roles[i]||fs::weakly_canonical(b.at("path").get<std::string>())!=fs::weakly_canonical(frames[i])||
             !read_file_snapshot(frames[i],bytes)||sha256_hex(bytes)!=b.at("sha256").get<std::string>()||
             bytes.size()!=b.at("bytes").get<std::size_t>()||!load_png(std::string_view(bytes),image))
            throw std::runtime_error("Actual A/B/empty source binding mismatch");
          if(!i){width=image.w;height=image.h;}else if(width!=image.w||height!=image.h)
            throw std::runtime_error("Actual source geometry changed");
          if(!width||!height||width>5120||height>5120)throw std::runtime_error("Invalid source geometry");
          inputs[i]=jp_texture(dev,width,height,DXGI_FORMAT_B8G8R8A8_UNORM,D3D11_BIND_SHADER_RESOURCE,image.bgra.data(),width*4);
          views[i]=inputs[i].srv.Get();
          upload_hashes[i]=sha256_hex(std::string_view(reinterpret_cast<const char *>(image.bgra.data()),image.bgra.size()));
          auto binding=b;binding["decoded_bgra8_sha256"]=upload_hashes[i];bindings.push_back(binding);
        }
        // All source decoding/allocation/hash work is complete before actual estimator timing.
        const auto result=probe::run(est,views,placement=="before-interop-unmaps"?
          models::joined_completion_gate_benchmark::placement_e::before_interop_unmaps:
          models::joined_completion_gate_benchmark::placement_e::after_interop_unmaps);
        // Genuine immutable GPU source readback, only AFTER release/completion. No CPU warp.
        for(std::size_t i=0;i<3;++i) {
          D3D11_TEXTURE2D_DESC d{};inputs[i].texture->GetDesc(&d);d.Usage=D3D11_USAGE_STAGING;
          d.BindFlags=d.MiscFlags=0;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;ComPtr<ID3D11Texture2D> stage;
          jp_require(dev->CreateTexture2D(&d,nullptr,&stage),"Source-check staging allocation failed");
          std::string raw(std::size_t(width)*height*4,'\0');
          ctx->CopyResource(stage.Get(),inputs[i].texture.Get());D3D11_MAPPED_SUBRESOURCE m{};
          jp_require(ctx->Map(stage.Get(),0,D3D11_MAP_READ,0,&m),"Source-check Map failed");
          for(UINT y=0;y<height;++y)std::memcpy(raw.data()+std::size_t(y)*width*4,
            static_cast<const char *>(m.pData)+std::size_t(y)*m.RowPitch,std::size_t(width)*4);
          ctx->Unmap(stage.Get(),0);const auto h=sha256_hex(raw);
          if(h!=upload_hashes[i])throw std::runtime_error("Actual immutable GPU source bytes differ from decoded input");
          bindings.at(i)["GPU_readback_bgra8_sha256"]=h;
        }
        const auto owner=[](const models::estimate_result &r,const models::host_sbs_gpu_completion_receipt::receipt_t &receipt) {
          return nlohmann::json{{"publication",jp_publication(r.publication)},
            {"receipt_expected",jp_publication(receipt.expected)},{"depth_owner_frame_id",receipt.depth_owner_frame_id},
            {"depth_owner_timestamp_us",receipt.depth_owner_timestamp_us},{"authenticated",models::parallax_v2_result_is_authenticated(r)}};
        };
        jp_json(root/"result.json",{{"schema",1},{"probe_kind","actual-pre-unmap-feasibility-v1"},
          {"plan_sha256",sha256_hex(plan_bytes)},{"gate_placement",placement},{"classification",result.classification},
          {"sources",bindings},{"source_geometry",{width,height}},{"prewarm",owner(result.warmup,result.warmup_receipt)},
          {"A",owner(result.A,result.A_receipt)},{"B",owner(result.B,result.B_receipt)},
          {"B_submit_begin_wall_us",result.B_submit_begin_us},{"B_submit_return_wall_us",result.B_submit_return_us},
          {"B_submit_host_duration_us",result.B_submit_return_us-result.B_submit_begin_us},
          {"before_B",pu_gate(result.before_B)},{"at_B_submit_return",pu_gate(result.at_submit_return)},
          {"first_real_poll",{{"begin_wall_us",result.first_poll_begin_us},{"return_wall_us",result.first_poll_return_us},
            {"ready",result.first_poll_ready},{"query_count",result.first_poll_query_count}}},
          {"after_first_real_poll",pu_gate(result.after_first_real_poll)},{"final_gate",pu_gate(result.final_gate)},
          {"host_callback_not_a_GPU_kernel_delay",true},{"common_tail_not_child_order_test",true},
          {"queued_continuation_or_D3D_draw_tested",false},{"performance_qualified",false},
          {"cue_or_EMPTY_authority",false},{"normal_joined_cleanup_completed",true}});
        return 0;
      }catch(const std::exception &e){
        est.release_joined_completion_gate_for_benchmark();
        if(created&&!fs::exists(root/"failure.json"))jp_json(root/"failure.json",{{"schema",1},{"error",e.what()},
          {"actual_gate",jp_gate(est.joined_completion_gate_status_for_benchmark())},{"measurement_valid",false}});
        BOOST_LOG(error)<<"sbs-bench pre-unmap feasibility: "<<e.what();return 12;
      }
    }
