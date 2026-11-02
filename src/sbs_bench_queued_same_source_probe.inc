// Private branch only; included after the existing jp_*/pu_gate helpers.
    int run_queued_same_source_probe(const opts &o,const std::vector<fs::path> &frames,
      const std::vector<std::uint64_t> &timestamps,ID3D11Device *dev,ID3D11DeviceContext *ctx,
      models::video_depth_estimator &est,ID3D11VertexShader *vs,ID3D11PixelShader *ordinary_ps,
      ID3D11SamplerState *sampler,int max_width,const std::string &plan_bytes) {
      namespace qrg=models::research_queued_render_guard_v1;
      namespace gate=models::joined_completion_gate_benchmark;
      namespace util=external_pre_unmap_feasibility_v1;
      const auto root=fs::path(o.out)/"queued-same-source-probe-v1";bool created=false;
      constexpr std::size_t slots=21,phases=5;
      const std::array<const char *,phases> phase_names{{"matched_A","queued_B_before_ACK","matched_B_after_ACK","matched_B_after_no_pending_finish","matched_empty_C"}};
      const std::array<const char *,slots> names{{"packed.bgra8","source.bgra8","Base.f32","state.u32","M.u32","display-source.rgba16f","SCP.u32","SCT.u32","transaction.u32","arguments.u32","contour-CB.u32","b2.u32","b3.u32","guard-CB.u32","depth-CB.u32","source-region-CB.u32","OCR.u32","UIP.u32","cut-state.u32","depth-frame-state.u32","ema-motion-mask.u32"}};
      struct capture_t {std::array<std::vector<char>,slots> bytes;std::array<bool,slots> copied{};nlohmann::json metadata;};
      std::array<jp_stage_t,slots> stages;std::array<capture_t,phases> captures;bool flushed=false;
      nlohmann::json diagnostic=nlohmann::json::object();
      const auto flush=[&](){
        if(flushed)return;flushed=true;
        for(std::size_t p=0;p<phases;++p){
          nlohmann::json artifacts=nlohmann::json::object();bool any=false;
          const auto folder=root/phase_names[p];
          for(std::size_t k=0;k<slots;++k)if(captures[p].copied[k]){
            if(!any){fs::create_directories(folder);any=true;}const auto path=folder/names[k];
            std::ofstream f(path,std::ios::binary);f.write(captures[p].bytes[k].data(),static_cast<std::streamsize>(captures[p].bytes[k].size()));f.flush();
            if(!f.good())throw std::runtime_error("Queued probe raw write failed");artifacts[names[k]]=jp_file(path);
          }
          if(any){auto j=captures[p].metadata;j["artifacts"]=artifacts;j["phase"]=phase_names[p];jp_json(folder/"phase.json",j);}
        }
      };
      try {
        const auto plan=nlohmann::json::parse(plan_bytes);const bool treatment=plan.at("queue_enabled").get<bool>();
        if(plan.at("schema")!=1||plan.at("probe_kind")!="queued-same-source-draw-v1"||frames.size()!=3||
          timestamps!=std::vector<std::uint64_t>{33334,66667,100000}||plan.at("observation_timestamps_us")!=nlohmann::json(timestamps)||
          plan.at("hold_after_entry_ms")!=120||plan.at("callback_safety_timeout_ms")!=500)throw std::runtime_error("Invalid queued probe protocol");
        if(fs::exists(root))throw std::runtime_error("Queued probe refuses overwrite");fs::create_directories(root);created=true;
        std::array<jp_texture_t,3> source;std::array<ID3D11ShaderResourceView *,3> input{};
        nlohmann::json bindings=nlohmann::json::array();UINT width=0,height=0;
        for(std::size_t i=0;i<3;++i){
          const auto &b=plan.at("sources").at(i);std::string bytes;rgba_image image;
          constexpr std::array<const char *,3> roles{{"A","B","empty"}};
          if(b.at("role")!=roles[i]||fs::weakly_canonical(b.at("path").get<std::string>())!=fs::weakly_canonical(frames[i])||
            !read_file_snapshot(frames[i],bytes)||sha256_hex(bytes)!=b.at("sha256").get<std::string>()||
            bytes.size()!=b.at("bytes").get<std::size_t>()||!load_png(std::string_view(bytes),image))throw std::runtime_error("Source binding mismatch");
          if(!i){width=image.w;height=image.h;}else if(width!=image.w||height!=image.h)throw std::runtime_error("Source geometry changed");
          if(!width||!height||width>5120||height>5120)throw std::runtime_error("Invalid source extent");
          source[i]=jp_texture(dev,width,height,DXGI_FORMAT_B8G8R8A8_UNORM,D3D11_BIND_SHADER_RESOURCE,image.bgra.data(),width*4);input[i]=source[i].srv.Get();
          auto actual=b;actual["decoded_bgra8_sha256"]=sha256_hex(std::string_view(reinterpret_cast<const char *>(image.bgra.data()),image.bgra.size()));bindings.push_back(actual);
        }
        constexpr std::array<models::host_sbs_shader_cache::shader_spec,1> specs{{{"sbs_reprojection_v2_live_ps.hlsl","main_ps","ps_5_0"}}};
        const auto guard_root=fs::path(plan.at("private_guard_shader_root").get<std::string>());
        const auto snapshot=models::host_sbs_shader_cache::snapshot_sources(guard_root,specs);
        if(!snapshot||models::host_sbs_shader_cache::source_closure_sha256(snapshot)!=plan.at("private_guard_source_closure_sha256").get<std::string>())throw std::runtime_error("QRG closure mismatch");
        const auto blob=models::host_sbs_shader_cache::get(snapshot,specs[0]);ComPtr<ID3D11PixelShader> guarded_ps;
        if(!blob)throw std::runtime_error("QRG PS compile failed");jp_require(dev->CreatePixelShader(blob->data(),blob->size(),nullptr,&guarded_ps),"QRG PS creation failed");
        std::string geometry_error;const auto output=detail::resolve_sbs_geometry(width,height,o.eye_w,o.eye_h,o.output_scale,max_width,geometry_error);
        if(!output)throw std::runtime_error("SBS geometry unavailable");
        auto packed=jp_texture(dev,output->sbs_width,output->sbs_height,DXGI_FORMAT_B8G8R8A8_UNORM,D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE);
        auto private_warm=jp_texture(dev,output->sbs_width,output->sbs_height,DXGI_FORMAT_B8G8R8A8_UNORM,D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE);
        ComPtr<ID3D11RenderTargetView> rtv,warm_rtv;jp_require(dev->CreateRenderTargetView(packed.texture.Get(),nullptr,&rtv),"RTV failed");jp_require(dev->CreateRenderTargetView(private_warm.texture.Get(),nullptr,&warm_rtv),"Warm RTV failed");
        const auto geometry=models::make_host_sbs_v2_full_frame_geometry(output->content_scale_x,output->content_scale_y);const auto b2=const_buffer(dev,geometry);
        ComPtr<ID3D11Buffer> b3,guard_cb;
        const auto make_cb=[&](UINT bytes,ComPtr<ID3D11Buffer> &target){D3D11_BUFFER_DESC d{};d.ByteWidth=bytes;d.Usage=D3D11_USAGE_DEFAULT;d.BindFlags=D3D11_BIND_CONSTANT_BUFFER;jp_require(dev->CreateBuffer(&d,nullptr,&target),"Mutable CB failed");};
        make_cb(32,b3);make_cb(sizeof(qrg::expected_t),guard_cb);
        const qrg::expected_t disabled_guard{};ctx->UpdateSubresource(guard_cb.Get(),0,nullptr,&disabled_guard,0,0);
        D3D11_VIEWPORT viewport{0,0,float(output->sbs_width),float(output->sbs_height),0,1};
        const auto resource=[](ID3D11ShaderResourceView *v){ComPtr<ID3D11Resource> r;if(v)v->GetResource(&r);return r;};
        const auto prepare=[&](std::size_t k,ID3D11Resource *source_resource,UINT fixed_buffer_bytes=0){
          auto &s=stages[k];if(source_resource)source_resource->GetType(&s.dimension);else s.dimension=D3D11_RESOURCE_DIMENSION_BUFFER;
          if(s.dimension==D3D11_RESOURCE_DIMENSION_TEXTURE2D){ComPtr<ID3D11Texture2D> t,stage;jp_require(source_resource->QueryInterface(IID_PPV_ARGS(&t)),"Texture QI failed");t->GetDesc(&s.texture);
            const auto format=s.texture.Format;const std::size_t pixel=format==DXGI_FORMAT_R16G16B16A16_FLOAT?8:4;
            if(format!=DXGI_FORMAT_R16G16B16A16_FLOAT&&format!=DXGI_FORMAT_R32_FLOAT&&format!=DXGI_FORMAT_R32_UINT&&format!=DXGI_FORMAT_B8G8R8A8_UNORM)throw std::runtime_error("Unsupported queued texture format");
            s.row_bytes=std::size_t(s.texture.Width)*pixel;s.rows=s.texture.Height;auto d=s.texture;d.Usage=D3D11_USAGE_STAGING;d.BindFlags=d.MiscFlags=0;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;jp_require(dev->CreateTexture2D(&d,nullptr,&stage),"Texture staging failed");jp_require(stage.As(&s.resource),"Texture staging QI failed");
          }else if(s.dimension==D3D11_RESOURCE_DIMENSION_BUFFER){D3D11_BUFFER_DESC d{};if(source_resource){ComPtr<ID3D11Buffer> b;jp_require(source_resource->QueryInterface(IID_PPV_ARGS(&b)),"Buffer QI failed");b->GetDesc(&d);}else d.ByteWidth=fixed_buffer_bytes;
            s.buffer_bytes=d.ByteWidth;s.row_bytes=d.ByteWidth;s.rows=1;d.Usage=D3D11_USAGE_STAGING;d.BindFlags=d.MiscFlags=d.StructureByteStride=0;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;ComPtr<ID3D11Buffer> stage;jp_require(dev->CreateBuffer(&d,nullptr,&stage),"Buffer staging failed");jp_require(stage.As(&s.resource),"Buffer staging QI failed");
          }else throw std::runtime_error("Unknown queued resource type");
          for(auto &c:captures)c.bytes[k].resize(s.row_bytes*s.rows);
        };
        const auto copy=[&](std::size_t p,std::size_t k,ID3D11Resource *r){
          auto &s=stages[k];auto &c=captures[p];if(!r||!s.resource||c.copied[k])throw std::runtime_error("Missing/duplicate queued capture");D3D11_RESOURCE_DIMENSION kind{};r->GetType(&kind);if(kind!=s.dimension)throw std::runtime_error("Capture type changed");
          if(kind==D3D11_RESOURCE_DIMENSION_BUFFER){ComPtr<ID3D11Buffer>b;jp_require(r->QueryInterface(IID_PPV_ARGS(&b)),"Capture buffer QI failed");D3D11_BUFFER_DESC d{};b->GetDesc(&d);if(d.ByteWidth!=s.buffer_bytes)throw std::runtime_error("Capture buffer extent changed");}
          else{ComPtr<ID3D11Texture2D>t;jp_require(r->QueryInterface(IID_PPV_ARGS(&t)),"Capture texture QI failed");D3D11_TEXTURE2D_DESC d{};t->GetDesc(&d);if(d.Width!=s.texture.Width||d.Height!=s.texture.Height||d.Format!=s.texture.Format||d.MipLevels!=1||d.ArraySize!=1||d.SampleDesc.Count!=1)throw std::runtime_error("Capture texture extent changed");}
          ctx->CopyResource(s.resource.Get(),r);D3D11_MAPPED_SUBRESOURCE m{};jp_require(ctx->Map(s.resource.Get(),0,D3D11_MAP_READ,0,&m),"Queued capture Map failed");
          for(std::size_t y=0;y<s.rows;++y)std::memcpy(c.bytes[k].data()+y*s.row_bytes,static_cast<const char *>(m.pData)+y*(kind==D3D11_RESOURCE_DIMENSION_TEXTURE2D?m.RowPitch:s.row_bytes),s.row_bytes);ctx->Unmap(s.resource.Get(),0);c.copied[k]=true;
        };
        const auto srv_copy=[&](std::size_t p,std::size_t k,ID3D11ShaderResourceView *v){auto r=resource(v);copy(p,k,r.Get());};
        const auto draw=[&](ID3D11ShaderResourceView *color,ID3D11ShaderResourceView *base,ID3D11ShaderResourceView *state,ID3D11ShaderResourceView *matte,ID3D11ShaderResourceView *display,ID3D11ShaderResourceView *scp,ID3D11ShaderResourceView *sct,
          std::uint64_t generation,ID3D11PixelShader *pixel,ID3D11RenderTargetView *target,ID3D11Buffer *expected,ID3D11ShaderResourceView *transaction){
          const models::host_sbs_subtitle_render_constants_t constants{.enabled=1u,.color_mode=0u,.analysis_generation_low=std::uint32_t(generation),.analysis_generation_high=std::uint32_t(generation>>32u),.source_left=0,.source_top=0,.source_width=width,.source_height=height};ctx->UpdateSubresource(b3.Get(),0,nullptr,&constants,0,0);
          const platf::dxgi::host_sbs_v2_draw_command_t command{.render_targets={target,nullptr},.render_target_count=1,.vertex_shader=vs,.pixel_shader=pixel,.viewport=viewport,.sampler=sampler,
            .shader_resources={color,base,state,matte,display,scp,sct},.geometry_constants=b2.Get(),.subtitle_constants=b3.Get()};
          qrg::scoped_binding_t binding;if(!binding.bind(ctx,expected,transaction)||!platf::dxgi::record_host_sbs_v2_draw(ctx,command))throw std::runtime_error("Guarded shared recorder failed");
        };
        const auto normal_draw=[&](ID3D11ShaderResourceView *color,const models::estimate_result &r,ID3D11PixelShader *pixel,ID3D11RenderTargetView *target){draw(color,r.shadow_base_final_parallax.Get(),r.shadow_state.Get(),r.subtitle_glyph_matte.Get(),r.subtitle_contour_source.Get(),r.subtitle_condition_params.Get(),r.subtitle_contour_identity.Get(),r.input_region.analysis_generation,pixel,target,nullptr,nullptr);};
        util::require(est.set_completed_ocr_conditioner_research_enabled(true),"Completed conditioner diagnostic flag unavailable");
        if(treatment)util::require(est.set_queued_same_source_gpu_continuation_research_enabled(true),"Queued API pre-input flag failed");
        gate::configuration_t configuration;configuration.enabled=true;configuration.placement=gate::placement_e::before_interop_unmaps;configuration.target_frame_id=3;configuration.hold_after_callback_enter_ms=120;configuration.callback_entry_timeout_ms=30000;configuration.callback_safety_timeout_ms=500;
        util::require(est.configure_joined_completion_gate_for_benchmark(configuration),"Pre-unmap gate configuration failed");
        const auto submit=[&](std::size_t i,std::uint64_t frame,std::uint64_t observation){const models::gpu_adaptive_reuse_request force{.observation_timestamp_us=observation};return est.estimate_depth(input[i],models::input_color_space::srgb,frame,false,{},models::depth_optional_work_mode_e::ordinary,force,0);};
        const auto finish=[&](std::uint64_t frame,std::uint64_t observation){auto r=est.finish_pending_depth_for_evaluation(models::input_color_space::srgb,false);util::require(r.completed_frame_valid&&r.completed_frame_id==frame&&r.publication.observation_timestamp_us==observation&&models::parallax_v2_result_is_authenticated(r),"Genuine exact-frame ACK failed");return r;};
        util::require(submit(2,1,1).inference_enqueued,"Warmup failed");auto warm=finish(1,1);const auto warm_receipt=util::receipt(est,warm);
        util::require(submit(0,2,33334).inference_enqueued,"A failed");auto A=finish(2,33334);const auto A_receipt=util::receipt(est,A);
        auto A_views=est.completed_ocr_conditioner_working_views_for_evaluation(A.publication);util::require(A_views.has_value(),"Genuine A diagnostic resources unavailable");
        const std::array<ID3D11ShaderResourceView *,slots> A_srv{{packed.srv.Get(),input[0],A.shadow_base_final_parallax.Get(),A.shadow_state.Get(),A.subtitle_glyph_matte.Get(),A.subtitle_contour_source.Get(),A.subtitle_condition_params.Get(),A.subtitle_contour_identity.Get(),nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,A.ocr_box_record.Get(),A.subtitle_locator_state.Get(),A.cut_state.Get(),A.depth_frame_state.Get(),A.ema_motion_mask.Get()}};
        // Storage-only handles: never reuse A's semantic clock/publication as B authority.
        // They may be sampled after release only if later actual ACK proves identical allocation.
        std::array<ComPtr<ID3D11Resource>,5> unpublished_storage;
        for(std::size_t k=0;k<unpublished_storage.size();++k)unpublished_storage[k]=resource(A_srv[16u+k]);
        for(std::size_t k=0;k<slots;++k)if(A_srv[k]){auto r=resource(A_srv[k]);prepare(k,r.Get());}
        prepare(8,A_views->transaction.Get());prepare(9,A_views->dispatch_arguments.Get());prepare(10,A_views->constants.Get());prepare(11,b2.Get());prepare(12,b3.Get());prepare(13,guard_cb.Get());prepare(14,nullptr,64);prepare(15,nullptr,16);
        // Real normal-A and disabled-private-PS draws/readbacks warm both shader paths before B.
        normal_draw(input[0],A,ordinary_ps,rtv.Get());srv_copy(0,0,packed.srv.Get());
        normal_draw(input[0],A,guarded_ps.Get(),warm_rtv.Get());srv_copy(3,0,private_warm.srv.Get());
        util::require(captures[0].bytes[0]==captures[3].bytes[0],"Disabled QRG warm-A differs from literal ordinary renderer");captures[3].copied[0]=false;
        const auto capture_normal=[&](std::size_t p,ID3D11ShaderResourceView *color,const models::estimate_result &r){
          auto views=est.completed_ocr_conditioner_working_views_for_evaluation(r.publication);util::require(views.has_value(),"Own completed diagnostic getter failed");
          const std::array<ID3D11ShaderResourceView *,slots> reads{{packed.srv.Get(),color,r.shadow_base_final_parallax.Get(),r.shadow_state.Get(),r.subtitle_glyph_matte.Get(),r.subtitle_contour_source.Get(),r.subtitle_condition_params.Get(),r.subtitle_contour_identity.Get(),nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,r.ocr_box_record.Get(),r.subtitle_locator_state.Get(),r.cut_state.Get(),r.depth_frame_state.Get(),r.ema_motion_mask.Get()}};
          for(std::size_t k=0;k<slots;++k)if(reads[k]&&!captures[p].copied[k])srv_copy(p,k,reads[k]);
          copy(p,8,views->transaction.Get());copy(p,9,views->dispatch_arguments.Get());copy(p,10,views->constants.Get());copy(p,11,b2.Get());copy(p,12,b3.Get());
          const auto receipt=util::receipt(est,r);captures[p].metadata={{"own_publication",jp_publication(r.publication)},{"own_receipt_expected",jp_publication(receipt.expected)},{"depth_owner_frame",receipt.depth_owner_frame_id},{"depth_owner_observation_us",receipt.depth_owner_timestamp_us},{"completed_authority",true}};
        };
        capture_normal(0,input[0],A);
        diagnostic={{"schema",1},{"queue_enabled",treatment},{"plan_sha256",sha256_hex(plan_bytes)},{"sources",bindings},{"prewarm_publication",jp_publication(warm.publication)},{"prewarm_receipt",jp_publication(warm_receipt.expected)},{"A_publication",jp_publication(A.publication)},
          {"all_staging_CPU_vectors_PS_CS_warm_before_B",true},{"guard_direct_unit_tests_qualified",false},{"performance_qualified",false}};
        diagnostic["source_width"]=width;diagnostic["source_height"]=height;diagnostic["packed_width"]=output->sbs_width;diagnostic["packed_height"]=output->sbs_height;
        diagnostic["field_width"]=A.field_width;diagnostic["field_height"]=A.field_height;
        diagnostic["layouts"]=nlohmann::json::array();
        for(std::size_t k=0;k<slots;++k){const auto &s=stages[k];diagnostic["layouts"].push_back({{"slot",k},{"name",names[k]},{"dimension",static_cast<unsigned>(s.dimension)},
          {"texture_width",s.texture.Width},{"texture_height",s.texture.Height},{"DXGI_format",static_cast<unsigned>(s.texture.Format)},
          {"buffer_bytes",s.buffer_bytes},{"row_bytes",s.row_bytes},{"rows",s.rows},{"bytes_per_phase",s.row_bytes*s.rows}});}
        std::optional<models::queued_same_source_gpu_continuation_research_views_t> ticket;
        ComPtr<ID3D11Buffer> queued_depth_storage,queued_source_storage;
        gate::snapshot_t before_queue,after_queue,after_record,after_record_query;
        std::int64_t queue_begin=0,queue_return=0,draw_begin=0,draw_return=0,flush_return=0;
        bool before_busy=false,after_busy=false,after_query_issued=false,old_A_current_after_queue=false,planned_current_after_queue=false;std::string queue_error;queue_error.reserve(256);
        const char *availability="ORDINARY_MATCHED_CONTROL";
        // B epoch starts here. No caller allocation/JIT/hash/JSON/Map until release.
        const auto epoch=util::now_us();
        try {
          util::require(submit(1,3,66667).inference_enqueued,"B failed");
          const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
          do{before_queue=util::status(est);if(before_queue.callback_entered)break;std::this_thread::yield();}while(std::chrono::steady_clock::now()<deadline);
          util::require(before_queue.callback_entered&&!before_queue.released,"Controlled B callback window unavailable");
          const auto before=est.try_finish_pending_depth_nonblocking(models::input_color_space::srgb,false);before_queue=util::status(est);
          before_busy=!before.ready&&before.query_count>0&&before_queue.last_actual_event_query==CUDA_ERROR_NOT_READY&&!before_queue.released;
          if(!before_busy){est.release_joined_completion_gate_for_benchmark();diagnostic["unavailable_before_queue"]=pu_gate(before_queue);diagnostic["before_query_ready"]=before.ready;throw std::runtime_error("No actual NOT_READY query before queued operation");}
          if(treatment){
            queue_begin=util::now_us();ticket=est.queue_pending_same_source_gpu_continuation_for_evaluation(input[1],3,66667,queue_error);queue_return=util::now_us();after_queue=util::status(est);
            util::require(ticket.has_value(),"Exact same-source queue API rejected B");
            const auto &q=*ticket;util::require(q.submitted_source.Get()==input[1]&&q.submitted_frame_id==3&&q.observation_timestamp_us==66667&&q.both_unmap_calls_succeeded&&q.postprocess_queued&&q.postprocess_request_flags==4&&q.transaction_token!=0&&q.ticket_sequence!=0,"Wrong queued ticket/source/context");
            old_A_current_after_queue=est.publication_is_current(A.publication);planned_current_after_queue=est.publication_is_current(q.planned_publication);
            util::require(!old_A_current_after_queue&&!planned_current_after_queue,"Queue exposed old or planned CPU publication authority");
            queued_depth_storage=q.depth_constants;queued_source_storage=q.source_region_constants;
            const auto expected=qrg::make_expected(q.transaction_token,q.expected_work,q.observation_timestamp_us);
            draw_begin=util::now_us();ctx->UpdateSubresource(guard_cb.Get(),0,nullptr,&expected,0,0);
            draw(q.submitted_source.Get(),q.base.Get(),q.state.Get(),q.glyph_matte.Get(),q.display_band.Get(),q.condition_params.Get(),q.contour_identity.Get(),q.input_region.analysis_generation,guarded_ps.Get(),rtv.Get(),guard_cb.Get(),q.transaction_raw.Get());
            draw_return=util::now_us();ctx->Flush();flush_return=util::now_us();after_record=util::status(est);
            if(!after_record.released){after_query_issued=true;const auto after=est.try_finish_pending_depth_nonblocking(models::input_color_space::srgb,false);after_record_query=util::status(est);after_busy=!after.ready&&after.query_count>0&&after_record_query.last_actual_event_query==CUDA_ERROR_NOT_READY&&!after_record_query.released;
              if(after.ready){est.release_joined_completion_gate_for_benchmark();diagnostic["availability"]="POST_RECORD_QUERY_ALREADY_CONSUMED_READY_B_NO_TICKET_READBACK";
                diagnostic["after_record"]=pu_gate(after_record);diagnostic["after_record_query"]=pu_gate(after_record_query);diagnostic["ready_query_actual_publication"]=jp_publication(after.result.publication);
                throw std::runtime_error("Post-record query consumed ready B; ticket lifetime expired, scheduling proof unavailable");}}
            if(after_queue.released)availability="QUEUE_API_RETURNED_AFTER_RELEASE";
            else if(after_record.released)availability="DRAW_OR_FLUSH_RETURNED_AFTER_RELEASE";
            else if(!after_busy)availability="NO_ACTUAL_NOT_READY_AFTER_RECORD";
            else availability="QUEUED_AND_GUARDED_DRAW_RECORDED_WHILE_ACTUAL_ROOT_NOT_READY";
          }
          // No readback until independent timer release; no forced early release on success.
          do{if(util::status(est).callback_exited)break;std::this_thread::yield();}while(std::chrono::steady_clock::now()<deadline);
          const auto released=util::status(est);util::require(released.callback_exited&&released.released&&!released.callback_timed_out&&!released.callback_entry_timed_out,"Actual independent release failed");
          if(ticket){const auto &q=*ticket;const std::array<ID3D11ShaderResourceView *,8> reads{{packed.srv.Get(),input[1],q.base.Get(),q.state.Get(),q.glyph_matte.Get(),q.display_band.Get(),q.condition_params.Get(),q.contour_identity.Get()}};
            for(std::size_t k=0;k<reads.size();++k)srv_copy(1,k,reads[k]);copy(1,8,q.transaction.Get());copy(1,9,q.dispatch_arguments.Get());copy(1,10,q.contour_constants.Get());copy(1,11,b2.Get());copy(1,12,b3.Get());copy(1,13,guard_cb.Get());copy(1,14,q.depth_constants.Get());copy(1,15,q.source_region_constants.Get());
            for(std::size_t k=0;k<unpublished_storage.size();++k)copy(1,16u+k,unpublished_storage[k].Get());
            captures[1].metadata={{"ticket_sequence",q.ticket_sequence},{"submitted_frame",q.submitted_frame_id},{"source_observation_us",q.observation_timestamp_us},{"transaction_token",q.transaction_token},{"expected_work",q.expected_work},{"postprocess_flags",q.postprocess_request_flags},{"planned_publication_not_completed",jp_publication(q.planned_publication)},{"completed_authority",false},{"capture_begin_after_callback_release",true},
              {"slots16_to20_unpublished_mutable_storage_only",true},{"old_A_packet_authority_claim",false},{"same_allocation_as_actual_ACK_verified",false}};
          }
          auto B=finish(3,66667);util::require(est.publication_is_current(B.publication),"Actual B ACK did not publish its own current key");normal_draw(input[1],B,ordinary_ps,rtv.Get());capture_normal(2,input[1],B);
          if(ticket){copy(2,14,queued_depth_storage.Get());copy(2,15,queued_source_storage.Get());}
          if(ticket){const std::array<ID3D11ShaderResourceView *,5> actual{{B.ocr_box_record.Get(),B.subtitle_locator_state.Get(),B.cut_state.Get(),B.depth_frame_state.Get(),B.ema_motion_mask.Get()}};
            for(std::size_t k=0;k<actual.size();++k){auto current=resource(actual[k]);util::require(current.Get()==unpublished_storage[k].Get(),"Unpublished diagnostic storage rotated before actual B ACK");}
            captures[1].metadata["same_allocation_as_actual_ACK_verified"]=true;}
          // Actual ACK consumed the pending unit. No-pending finish must return no NEW completion.
          const auto no_pending=est.finish_pending_depth_for_evaluation(models::input_color_space::srgb,false);
          util::require(!no_pending.completed_frame_valid&&!no_pending.inference_enqueued&&!no_pending.gpu_undecided_transaction_enqueued&&
            no_pending.publication==models::host_sbs_gpu_completion_receipt::expected_t{}&&est.publication_is_current(B.publication),"No-pending finish created completion/publication or invalidated B");
          normal_draw(input[1],B,ordinary_ps,rtv.Get());capture_normal(3,input[1],B);
          // Retained physical CB storage is diagnostic only after the ticket lease expires at ACK.
          if(ticket){copy(3,14,queued_depth_storage.Get());copy(3,15,queued_source_storage.Get());}
          for(std::size_t k=0;k<slots;++k)if(captures[2].copied[k])util::require(captures[3].copied[k]&&captures[2].bytes[k]==captures[3].bytes[k],"Second ACK mutated completed B data/render");
          if(ticket){for(std::size_t k=0;k<=12;++k)util::require(captures[1].bytes[k]==captures[2].bytes[k],"Queued B changed on ACK or differs from completed normal renderer");
            util::require(captures[1].bytes[14]==captures[2].bytes[14]&&captures[1].bytes[15]==captures[2].bytes[15],"ACK changed queued actual CBs");
            for(std::size_t k=16;k<slots;++k)util::require(captures[1].bytes[k]==captures[2].bytes[k],"ACK changed unpublished OCR/UIP/cut/minmaxEMA/motion storage");
            util::require(ticket->planned_publication==B.publication,"Planned queued identity differs from own genuine ACK");}
          util::require(submit(2,4,100000).inference_enqueued,"C failed");auto C=finish(4,100000);normal_draw(input[2],C,ordinary_ps,rtv.Get());capture_normal(4,input[2],C);
          const auto &m=captures[4].bytes[4];util::require(std::all_of(m.begin(),m.end(),[](char x){return x==0;}),"Genuine current empty C did not clear M");
          diagnostic["B_epoch_wall_us"]=epoch;diagnostic["availability"]=availability;diagnostic["before_queue_actual_busy"]=before_busy;diagnostic["after_record_actual_busy"]=after_busy;diagnostic["after_record_query_issued"]=after_query_issued;
          diagnostic["old_A_current_after_queue"]=old_A_current_after_queue;diagnostic["planned_current_after_queue"]=planned_current_after_queue;
          diagnostic["before_queue"]=pu_gate(before_queue);if(treatment){diagnostic["after_queue"]=pu_gate(after_queue);diagnostic["after_record"]=pu_gate(after_record);diagnostic["after_record_query"]=pu_gate(after_record_query);}
          diagnostic["queue_begin_wall_us"]=queue_begin;diagnostic["queue_return_wall_us"]=queue_return;diagnostic["draw_begin_wall_us"]=draw_begin;diagnostic["draw_return_wall_us"]=draw_return;diagnostic["flush_return_wall_us"]=flush_return;
          diagnostic["final_gate"]=pu_gate(util::status(est));diagnostic["B_publication"]=jp_publication(B.publication);diagnostic["C_publication"]=jp_publication(C.publication);
          diagnostic["no_pending_finish_completed_frame_valid"]=no_pending.completed_frame_valid;diagnostic["no_pending_finish_publication"]=jp_publication(no_pending.publication);
          diagnostic["B_full_bytes_unchanged_after_no_pending_finish"]=true;diagnostic["current_model_C_M_literal_zero"]=true;diagnostic["semantic_current_cue_or_EMPTY_authority_from_ticket"]=false;
          est.release_joined_completion_gate_for_benchmark();flush();jp_json(root/"result.json",diagnostic);return 0;
        }catch(...){est.release_joined_completion_gate_for_benchmark();try{flush();}catch(...){}throw;}
      }catch(const std::exception &e){est.release_joined_completion_gate_for_benchmark();if(created&&!fs::exists(root/"failure.json")){diagnostic["error"]=e.what();diagnostic["pass"]=false;jp_json(root/"failure.json",diagnostic);}BOOST_LOG(error)<<"queued same-source probe: "<<e.what();return 12;}
    }
