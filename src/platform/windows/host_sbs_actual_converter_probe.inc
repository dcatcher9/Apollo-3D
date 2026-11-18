// Included only inside platf::dxgi AFTER the actual converter class is complete.
// No duplicate converter policy, CUDA verdict, CPU warp or encoder session.
namespace actual_converter_probe_v1 {
  namespace {
    using Microsoft::WRL::ComPtr;
    using json = nlohmann::json;
    using clock_t = std::chrono::steady_clock;
    constexpr std::size_t phase_count=6, slot_count=51;
    constexpr std::array<const char *,phase_count> phases{{"matched_A","before_queue_B_old_working","queued_B_before_ACK","B_after_genuine_ACK","C_arrival_while_B_pending","matched_C"}};
    constexpr std::array<const char *,slot_count> names{{"draw-source.bgra8","Base.f32","state.u32","M.u32","display-source.rgba16f","SCP.u32","SCT.u32","OCR.u32","UIP.u32","depth-health.u32","frame-state.u32","EMA.u32","b2.u32","b3.u32","b5.u32","transaction.u32","arguments.u32","depth-CB.u32","region-CB.u32","contour-CB.u32","packed.bgra8","rgb-target.bgra8","current-input.bgra8","selected-M.u32","selected-receipt.u32","actual-draw-M.u32","baseline-body.u32","baseline-M.u32","numeric-proof.u32","current-P.f32","current-S.u32","actual-contour-CB.u32","temporal-M.u32","temporal-receipt.u32","actual-Frame256.u32","background-field.rgba32f","background-proof.u32","background-expected.u32","background-source-C24.u32","source5-analysis-band.rgba16f","CBC-body.u32","CBC-meta.u32","CBC-all-body-bits.u32","CBC-selected-bits.u32","CBC-cover-bits.u32","raw-M.u32","raw-meta.u32","F2-calibration.u32","F2-passable-root-cache.u32","stability-header160.u32","stability-image-RGB24-LOW8.u32"}};
    void require(bool value,const char *message) { if (!value) throw std::runtime_error(message); }
    void hr(HRESULT value,const char *message) { require(SUCCEEDED(value),message); }
    std::int64_t us() { return std::chrono::duration_cast<std::chrono::microseconds>(clock_t::now().time_since_epoch()).count(); }
    json binding(const std::filesystem::path &p) {
      const auto hash=models::file_sha256_hex(p);require(!hash.empty(),"Artifact hash failed");
      return {{"path",std::filesystem::absolute(p).string()},{"sha256",hash},{"bytes",std::filesystem::file_size(p)}};
    }
    void write_json(const std::filesystem::path &p,const json &j) {
      require(!std::filesystem::exists(p),"Probe refuses JSON overwrite");
      std::ofstream f(p);f<<j.dump(2)<<'\n';f.flush();require(f.good(),"JSON write failed");
    }
    json pub(const models::host_sbs_gpu_completion_receipt::expected_t &p) {
      return {{"estimator_generation",p.estimator_generation},{"publication_sequence",p.publication_sequence},{"frame_id",p.frame_id},
        {"observation_us",p.observation_timestamp_us},{"analysis_generation",p.analysis_generation},{"token",p.transaction_token},
        {"flags",p.flags},{"expected_work",p.expected_work},{"domain_tag",p.domain_tag},{"subtitle_domain",p.subtitle_domain},
        {"baseline_frame_id",p.baseline_frame_id},{"raw_coordinate_scale",p.raw_coordinate_scale},{"joint_plane_mode",p.joint_plane_mode},
        {"width",p.width},{"height",p.height},{"submission_class",static_cast<unsigned>(p.submission_class)},{"host_subtitle_outcome",static_cast<unsigned>(p.host_subtitle_outcome)}};
    }
    json gate_json(const std::optional<models::joined_completion_gate_benchmark::snapshot_t> &s) {
      if (!s) return nullptr;const auto &q=*s;
      return {{"target_frame",q.configuration.target_frame_id},{"entered",q.callback_entered},{"released",q.released},{"exited",q.callback_exited},
        {"entered_us",q.entered_wall_us},{"released_us",q.released_wall_us},{"exited_us",q.exited_wall_us},{"cancelled",q.cancelled},
        {"callback_timeout",q.callback_timed_out},{"entry_timeout",q.callback_entry_timed_out},{"capture_checked",q.capture_state_checked},
        {"capture_query_result",q.capture_state_query_result},{"capture_state",q.capture_state},
        {"last_actual_query",q.last_actual_event_query},{"query_count",q.actual_event_queries},{"not_ready_count",q.actual_not_ready_queries},
        {"ready_count",q.actual_ready_queries},{"failed_count",q.actual_failed_queries},
        {"post_unmap_query_recorded",q.post_unmap_query_recorded},{"post_unmap_query",q.post_unmap_query_result},
        {"ocr_unmap",{{"issued",q.optional_ocr_unmap.issued},{"result",q.optional_ocr_unmap.result},{"begin_us",q.optional_ocr_unmap.begin_wall_us},{"return_us",q.optional_ocr_unmap.return_wall_us}}},
        {"depth_unmap",{{"issued",q.depth_unmap.issued},{"result",q.depth_unmap.result},{"begin_us",q.depth_unmap.begin_wall_us},{"return_us",q.depth_unmap.return_wall_us}}}};
    }
    json event_json(const event_t &e) {
      return {{"kind",static_cast<unsigned>(e.kind)},{"conversion",e.conversion},{"model_frame",e.model_frame},{"observation_us",e.observation_us},
        {"host_event_us",e.host_event_us},{"copied_at_ticks",e.copied_at_ticks},{"copied_at_us",e.copied_at_us},
        {"source_stamp_ticks",e.source_stamp_ticks},{"content_stamp_ticks",e.content_stamp_ticks},
        {"current_source_stamp_ticks",e.current_source_stamp_ticks},{"current_content_stamp_ticks",e.current_content_stamp_ticks},
        {"context_generation",e.context_generation},{"transaction_token",e.transaction_token},{"request_flags",e.request_flags},
        {"input_region",{{"source_width",e.region.source_width},{"source_height",e.region.source_height},{"left",e.region.left},{"top",e.region.top},{"right",e.region.right},{"bottom",e.region.bottom},{"analysis_generation",e.region.analysis_generation},{"authority",static_cast<unsigned>(e.region.authority)},
          {"tensor_content",{e.region.tensor_content.left,e.region.tensor_content.top,e.region.tensor_content.right,e.region.tensor_content.bottom}}}},
        {"field_width",e.field_width},{"field_height",e.field_height},{"field_content",{e.field_content.left,e.field_content.top,e.field_content.right,e.field_content.bottom}},
        {"force_infer",e.force_infer},{"gpu_undecided",e.gpu_undecided},{"optional_work",e.optional_work},{"ddup_proof",e.ddup_proof},
        {"queued",e.queued},{"flat",e.flat},{"repeat",e.repeat},{"cached_current_colour",e.cached_current_colour},
        {"native_planes_written",e.native_planes_written},
        {"p010_chroma_tiled",e.p010_chroma_tiled},
        {"slot_frames",e.slot_frames},{"slot_pending",e.slot_pending},{"slot_observations",e.slot_observations},{"slot_contexts",e.slot_contexts},
        {"slot_source_addresses",e.slot_source_addresses},{"selected_slot",e.selected_slot},{"pending_slots",e.pending_slots},
        {"publication_current",e.publication_current},{"ticket_publication_current",e.ticket_publication_current},{"publication",pub(e.publication)},
        {"planned_publication_VALUE_ONLY",pub(e.planned_publication)},{"actual_draw_pixel_source",e.actual_draw_pixel_source},
        {"pixel_source_frame",e.pixel_source_frame},{"pixel_source_stamp_ticks",e.pixel_source_stamp_ticks},{"packed_valid",e.packed_valid},
        {"packed_age_us",e.packed_age_us},{"route_matches",e.route_matches},{"source_matches",e.source_matches},
        {"watermark_before",e.watermark_before},{"watermark_after",e.watermark_after},
        {"literal_query_observed",e.query_observed},{"literal_last_query_result",e.literal_last_query_result},{"literal_query_count",e.literal_query_count},
        {"literal_query_target_frame",e.literal_query_target_frame},
        {"literal_not_ready_count",e.literal_not_ready_count},{"literal_ready_count",e.literal_ready_count},{"gate_entered",e.gate_entered},{"gate_released",e.gate_released},
        {"selected_draw",e.selected_draw},{"selected_baseline_current",e.selected_baseline_current},{"selected_temporal_commands",e.selected_temporal_commands},{"selected_source_certificate_valid",e.selected_source_certificate_valid},{"actual_draw_selected_receipt_address",e.actual_draw_selected_receipt_address},
        {"background",{{"queue_issued",e.background.queue_issued},{"role_matched",e.background.role_matched},{"bound",e.background.bound},{"ps_observed_after_draw",e.background.ps_observed_after_draw},
          {"owner_serial",e.background.owner_serial},{"transport_serial",e.background.transport_serial},{"context_address",e.background.context_address},{"pixel_shader_address",e.background.pixel_shader_address},
          {"colour_key",e.background.colour_key},{"actual_role_pointers",e.background.actual_role_pointers},{"analysis_shape",e.background.analysis_shape},{"pixel_shader_closure",e.background.pixel_shader_closure}}},
        {"shared_warp",{{"selected",e.shared_warp.selected},{"queued",e.shared_warp.queued},
          {"role_matched",e.shared_warp.role_matched},{"bindings_observed",e.shared_warp.bindings_observed},
          {"no_background",e.shared_warp.no_background},{"serial",e.shared_warp.serial},
          {"pixel_shader_address",e.shared_warp.pixel_shader_address},{"shape",e.shared_warp.shape},
          {"source_closure",e.shared_warp.source_closure},
          {"weights_address",reinterpret_cast<std::uintptr_t>(e.shared_warp.weights.Get())},
          {"field_address",reinterpret_cast<std::uintptr_t>(e.shared_warp.field.Get())},
          {"rows_address",reinterpret_cast<std::uintptr_t>(e.shared_warp.rows.Get())},
          {"geometry_address",reinterpret_cast<std::uintptr_t>(e.shared_warp.geometry.Get())}}}};
    }
    struct fixture_display_t final : display_vram_t {
      bool declared_hdr=false;
      bool is_hdr() override { return declared_hdr; } // Explicit fixture transfer, no fake IDXGIOutput
      capture_e snapshot(const pull_free_image_cb_t &,std::shared_ptr<platf::img_t> &,std::chrono::milliseconds,bool) override { return capture_e::error; }
      capture_e release_snapshot() override { return capture_e::error; }
    };
    struct stage_t {
      ComPtr<ID3D11Resource> staging;
      D3D11_RESOURCE_DIMENSION kind{};D3D11_TEXTURE2D_DESC td{};UINT buffer_bytes=0,source_stride=0,source_bind=0,source_misc=0;
      std::size_t row_bytes=0,rows=0;std::vector<char> bytes;bool copied=false;
      std::int64_t copy_issued_us=0,read_complete_us=0;
      std::uintptr_t actual_resource_address=0;
      void prepare_buffer(ID3D11Device *d,UINT count) {
        D3D11_BUFFER_DESC b{};b.ByteWidth=count;b.Usage=D3D11_USAGE_STAGING;b.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
        ComPtr<ID3D11Buffer> out;hr(d->CreateBuffer(&b,nullptr,&out),"Staging buffer allocation failed");staging=out;
        kind=D3D11_RESOURCE_DIMENSION_BUFFER;buffer_bytes=count;row_bytes=count;rows=1;bytes.resize(row_bytes);
      }
      void prepare(ID3D11Device *d,ID3D11Resource *r) {
        require(r,"Missing prepare resource");r->GetType(&kind);
        if(kind==D3D11_RESOURCE_DIMENSION_BUFFER){ComPtr<ID3D11Buffer>b;hr(r->QueryInterface(IID_PPV_ARGS(&b)),"Buffer QI failed");D3D11_BUFFER_DESC bd{};b->GetDesc(&bd);prepare_buffer(d,bd.ByteWidth);}
        else {require(kind==D3D11_RESOURCE_DIMENSION_TEXTURE2D,"Unsupported resource");ComPtr<ID3D11Texture2D>t;hr(r->QueryInterface(IID_PPV_ARGS(&t)),"Texture QI failed");t->GetDesc(&td);
          require(td.MipLevels==1&&td.ArraySize==1&&td.SampleDesc.Count==1,"Unsupported texture layout");
          const bool nv12 = td.Format == DXGI_FORMAT_NV12;
          const bool p010 = td.Format == DXGI_FORMAT_P010;
          const UINT pixel=nv12?1u:p010?2u:td.Format==DXGI_FORMAT_R32G32B32A32_FLOAT?16u:td.Format==DXGI_FORMAT_R16G16B16A16_FLOAT?8u:4u;
          require(nv12||p010||pixel==16u||pixel==8u||td.Format==DXGI_FORMAT_R32_FLOAT||td.Format==DXGI_FORMAT_R32_UINT||td.Format==DXGI_FORMAT_B8G8R8A8_UNORM,"Unsupported staging format");
          auto desc=td;desc.Usage=D3D11_USAGE_STAGING;desc.BindFlags=0;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;desc.MiscFlags=0;
          ComPtr<ID3D11Texture2D>out;hr(d->CreateTexture2D(&desc,nullptr,&out),"Staging texture allocation failed");staging=out;
          // NV12 maps Y and interleaved UV consecutively with the same row pitch.
          row_bytes=std::size_t(td.Width)*pixel;rows=td.Height+((nv12||p010)?td.Height/2u:0u);bytes.resize(row_bytes*rows);}
      }
      bool copy(ID3D11DeviceContext *ctx,ID3D11Resource *r) noexcept {
        if(!r||!staging||copied)return false;D3D11_RESOURCE_DIMENSION k{};r->GetType(&k);if(k!=kind)return false;
        if(k==D3D11_RESOURCE_DIMENSION_BUFFER){ComPtr<ID3D11Buffer>b;if(FAILED(r->QueryInterface(IID_PPV_ARGS(&b))))return false;D3D11_BUFFER_DESC bd{};b->GetDesc(&bd);if(bd.ByteWidth!=buffer_bytes)return false;source_stride=bd.StructureByteStride;source_bind=bd.BindFlags;source_misc=bd.MiscFlags;}
        else {ComPtr<ID3D11Texture2D>t;if(FAILED(r->QueryInterface(IID_PPV_ARGS(&t))))return false;D3D11_TEXTURE2D_DESC d{};t->GetDesc(&d);if(d.Width!=td.Width||d.Height!=td.Height||d.Format!=td.Format||d.MipLevels!=1||d.ArraySize!=1||d.SampleDesc.Count!=1)return false;source_stride=0;source_bind=d.BindFlags;source_misc=d.MiscFlags;}
        actual_resource_address=reinterpret_cast<std::uintptr_t>(r);copy_issued_us=us();ctx->CopyResource(staging.Get(),r);copied=true;return true;
      }
      void read(ID3D11DeviceContext *ctx) {
        if(!copied)return;D3D11_MAPPED_SUBRESOURCE m{};hr(ctx->Map(staging.Get(),0,D3D11_MAP_READ,0,&m),"Staging Map failed");
        for(std::size_t y=0;y<rows;++y)std::memcpy(bytes.data()+y*row_bytes,static_cast<const char*>(m.pData)+y*(kind==D3D11_RESOURCE_DIMENSION_BUFFER?row_bytes:m.RowPitch),row_bytes);
        ctx->Unmap(staging.Get(),0);read_complete_us=us();
      }
    };
    ComPtr<ID3D11Resource> resource(ID3D11ShaderResourceView *v) {ComPtr<ID3D11Resource>r;if(v)v->GetResource(&r);return r;}
    struct recorder_t {
      ID3D11DeviceContext *ctx=nullptr;
      std::array<event_t,128> events{};std::size_t count=0;bool overflow=false,capture_fault=false;
      std::array<std::array<stage_t,slot_count>,phase_count> stages;
      std::array<bool,phase_count> captured{};
      std::array<event_t,phase_count> phase_event{};
      std::array<ComPtr<ID3D11ShaderResourceView>,5> old_working;
      int armed=-1;bool prepared=false,pending_arrival=false,selected_enabled=false,selected_outline=false,background_enabled=false;
      std::array<ComPtr<ID3D11Buffer>,3> queued_B_constant_aliases;
      std::uint64_t borrowed_draw_conversion = 0;
      std::array<stage_t, 2> borrowed_draw_sources;
      void capture(std::size_t phase,const event_t &e) noexcept {
        if(!prepared||captured[phase]){capture_fault=true;return;}
        auto copy=e;
        if(copy.repeat){for(auto &v:copy.views)v.Reset();for(auto &b:copy.buffers)b.Reset();}
        else for(std::size_t i=7;i<12;++i)if(!copy.views[i])copy.views[i]=old_working[i-7];
        if(phase==2)for(std::size_t i=0;i<3;++i)queued_B_constant_aliases[i]=copy.buffers[5+i];
        if(phase==3)for(std::size_t i=0;i<3;++i)if(!copy.buffers[5+i])copy.buffers[5+i]=queued_B_constant_aliases[i];
        for(std::size_t k=0;k<12;++k){const auto r=resource(copy.views[k].Get());if(r&&!stages[phase][k].copy(ctx,r.Get()))capture_fault=true;}
        for(std::size_t k=0;k<8;++k)if(copy.buffers[k]&&!stages[phase][12+k].copy(ctx,copy.buffers[k].Get()))capture_fault=true;
        if(copy.packed&&!stages[phase][20].copy(ctx,copy.packed.Get()))capture_fault=true;
        const auto current=resource(copy.current_input.Get());if(current&&!stages[phase][22].copy(ctx,current.Get()))capture_fault=true;
        if(phase!=1u&&phase!=4u&&!copy.repeat){
          constexpr std::array<std::size_t,11> map{23u,24u,25u,26u,27u,28u,29u,30u,32u,33u,34u};
          for(std::size_t k=0;k<map.size();++k){const auto r=resource(copy.selected_views[k].Get());if(r&&!stages[phase][map[k]].copy(ctx,r.Get()))capture_fault=true;}
          if(copy.selected_contour_constants&&!stages[phase][31u].copy(ctx,copy.selected_contour_constants.Get()))capture_fault=true;
          for(std::size_t k=0;k<copy.current_outline_stages.size();++k) {
            const auto r=resource(copy.current_outline_stages[k].Get());
            if(r&&!stages[phase][39u+k].copy(ctx,r.Get()))capture_fault=true;
          }
        }
        if(background_enabled&&(phase==0u||phase==3u||phase==5u)&&!copy.repeat){
          const auto&b=copy.background;
          if(!b.queue_issued||!b.role_matched||!b.bound||!b.ps_observed_after_draw||!b.field||!b.proof||!b.expected||!b.source_c24||b.colour_key[0]!=copy.conversion)capture_fault=true;
          const auto field=resource(b.field.Get()),proof=resource(b.proof.Get());
          if(!field||!proof||!stages[phase][35].copy(ctx,field.Get())||!stages[phase][36].copy(ctx,proof.Get())||!stages[phase][37].copy(ctx,b.expected.Get())||!stages[phase][38].copy(ctx,b.source_c24.Get()))capture_fault=true;
        }
        phase_event[phase]=copy;captured[phase]=true;
      }
      static void record(void *state,const event_t &e) noexcept {
        auto &s=*static_cast<recorder_t*>(state);if(s.count==s.events.size()){s.overflow=true;return;}s.events[s.count++]=e;
        if (s.borrowed_draw_conversion == e.conversion && e.kind == event_e::draw) {
          // The converter still owns both keys here. COM references alone do not pin the pixels.
          const auto matched = resource(e.views[0].Get());
          const auto current = resource(e.current_input.Get());
          if (!s.borrowed_draw_sources[0].copy(s.ctx, matched.Get()) ||
              !s.borrowed_draw_sources[1].copy(s.ctx, current.Get())) {
            s.capture_fault = true;
          }
        }
        if(e.kind==event_e::draw&&e.publication_current&&!e.queued)for(std::size_t i=7;i<12;++i)s.old_working[i-7]=e.views[i];
        if(s.armed==1&&e.kind==event_e::queue_begin)s.capture(1,e);
        if(s.armed==1&&e.kind==event_e::draw&&e.queued)s.capture(2,e);
        if(s.armed==3&&e.publication_current&&!e.queued&&e.publication.frame_id==3&&
           ((s.pending_arrival&&e.kind==event_e::ack)||(!s.pending_arrival&&e.kind==event_e::draw))){s.capture(3,e);if(s.pending_arrival)s.armed=5;}
        if(s.armed==4&&e.conversion==4&&(e.kind==event_e::packed_repeat||e.kind==event_e::draw))s.capture(4,e);
        if(s.armed==5&&e.kind==event_e::draw&&e.publication_current&&!e.queued&&e.publication.frame_id>3)s.capture(5,e);
      }
      const event_t *last_draw() const {for(std::size_t i=count;i>0;--i)if(events[i-1].kind==event_e::draw)return &events[i-1];return nullptr;}
      void prepare(ID3D11Device *device,const event_t &a,ID3D11Texture2D *target) {
        for(auto &phase:stages){
          for(std::size_t k=0;k<12;++k){auto r=resource(a.views[k].Get());require(r!=nullptr,"A working view missing");phase[k].prepare(device,r.Get());}
          for(std::size_t k=0;k<8;++k){if(a.buffers[k])phase[12+k].prepare(device,a.buffers[k].Get());
            else if(k==2)phase[14].prepare_buffer(device,48u);
            else if(k>=5)phase[12+k].prepare_buffer(device,k==5?64u:k==6?16u:96u);
            else throw std::runtime_error("A mandatory buffer missing");}
          phase[20].prepare(device,a.packed.Get());phase[21].prepare(device,target);
          auto current=resource(a.current_input.Get());require(current!=nullptr,"A current input missing");phase[22].prepare(device,current.Get());
          if(selected_enabled){
            for(std::size_t k=0;k<8u;++k){if(k==2u || (selected_outline&&k>=3u&&k<=5u))continue;const auto r=resource(a.selected_views[k].Get());require(r!=nullptr,"Selected A required diagnostic missing");constexpr std::array<std::size_t,8> map{23u,24u,25u,26u,27u,28u,29u,30u};phase[map[k]].prepare(device,r.Get());}
            auto m=resource(a.selected_views[0].Get());phase[25u].prepare(device,m.Get());phase[32u].prepare(device,m.Get());
            if(a.selected_contour_constants)phase[31u].prepare(device,a.selected_contour_constants.Get());else {require(background_enabled,"Selected A actual contourCB missing");phase[31u].prepare_buffer(device,96u);}phase[33u].prepare_buffer(device,1024u);phase[34u].prepare_buffer(device,256u);
          for(std::size_t k=0;k<a.current_outline_stages.size();++k) {
            const auto r=resource(a.current_outline_stages[k].Get());
            if(r)phase[39u+k].prepare(device,r.Get());
          }
          }

          if(background_enabled){const auto&b=a.background;require(b.queue_issued&&b.role_matched&&b.bound&&b.ps_observed_after_draw&&b.field&&b.proof&&b.expected&&b.source_c24,"A actual BG draw binding absent");const auto f=resource(b.field.Get()),p=resource(b.proof.Get());phase[35].prepare(device,f.Get());phase[36].prepare(device,p.Get());phase[37].prepare(device,b.expected.Get());phase[38].prepare(device,b.source_c24.Get());}
        }prepared=true;capture(0,a);require(!capture_fault,"A capture setup failed");
      }
      void copy_target(std::size_t phase,ID3D11Texture2D *target){require(stages[phase][21].copy(ctx,target),"RGB target copy failed");}
      void read_all(){for(auto &p:stages)for(auto &s:p)s.read(ctx);}
    };
  }

  int run_one(const request_t &r,bool pending_arrival,std::string &error,int native_codec = 1) noexcept {
    error.clear();std::shared_ptr<fixture_display_t> display;
    // The converter's borrowed output owner must survive converter teardown.
    std::unique_ptr<nvenc::nvenc_d3d11_native> native_encoder;
    d3d_base_encode_device converter;
    recorder_t records;ComPtr<ID3D11Texture2D> target;ComPtr<ID3D11RenderTargetView> target_rtv;
    std::array<std::shared_ptr<platf::img_t>,3> images;
    models::video_depth_estimator *est=nullptr;
    std::jthread releaser;std::atomic<bool> release_after_draw_query{false},timeout_forced_release{false};
    records.pending_arrival=pending_arrival;records.selected_enabled=r.selected_matte.has_value();
    records.selected_outline=r.selected_matte&&r.selected_matte->current_outline;records.background_enabled=r.background_observer;
    json result,adapter_entries=json::array(),selected_adapter=nullptr,capture_device_adapter=nullptr,converter_device_adapter=nullptr;bool created=false,released_before_readback=false;
    const auto *native_nv12_opt_in = std::getenv("SUNSHINE_TEST_HOST_SBS_NATIVE_NV12");
    const bool native_nv12 = r.packed_continuity && native_nv12_opt_in && std::string_view {native_nv12_opt_in} == "1";
    const auto *recycle_opt_in = std::getenv("SUNSHINE_TEST_HOST_SBS_BORROWED_CAPTURE_RECYCLE");
    const bool borrowed_recycle = r.packed_continuity && (r.hdr_input || (recycle_opt_in && std::string_view {recycle_opt_in} == "1"));
    const bool check_retained_rgb = borrowed_recycle && !native_nv12 && !r.native_p010;
    const auto *reference_opt_in = std::getenv("SUNSHINE_TEST_HOST_SBS_SHUTTER_COPY_REFERENCE");
    const bool copy_reference = reference_opt_in && std::string_view {reference_opt_in} == "1";
    const auto *cached_route_opt_in = std::getenv("SUNSHINE_TEST_HOST_SBS_CACHED_CURRENT_ROUTE");
    const bool cached_route = cached_route_opt_in && std::string_view {cached_route_opt_in} == "1";
    const auto root=cached_route ? r.output/"actual-converter-cached-current-route-v1" : r.native_p010 ? r.output/"actual-converter-native-p010-v1"/(native_codec == 1 ? "hevc" : "av1") :
      borrowed_recycle?r.output/"actual-converter-borrowed-capture-recycle-v1":
      r.packed_continuity?r.output/(native_nv12 ? "actual-converter-packed-continuity-nv12-v1" : "actual-converter-packed-continuity-v1"):
      r.output/(pending_arrival?"trajectory2-pending-C":"trajectory1-B-parity")/"actual-converter-rgb-probe-v1";
    const auto release=[&](){if(est)est->release_joined_completion_gate_for_benchmark();release_after_draw_query.store(true);if(releaser.joinable()){releaser.request_stop();releaser.join();}};
    try {
      require(r.enabled,"Probe not enabled");require(!std::filesystem::exists(root),"Probe refuses overwrite");
      require(!cached_route || (r.packed_continuity && !r.queued && !native_nv12 && !r.native_p010 &&
        !pending_arrival && !r.selected_matte && !r.temporal_source_bridge && !r.background_observer),
        "Cached-current route regression requires ordinary SDR/HDR RGB continuity");
      require(!r.native_p010 || (r.hdr_input && !native_nv12 && r.packed_continuity &&
        (native_codec == 1 || native_codec == 2) &&
        (r.p010_chroma_mode == "default" || r.p010_chroma_mode == "raster" || r.p010_chroma_mode == "no-uav")),
        "Invalid native P010 fixture declaration");
      require(!r.packed_continuity || (!r.queued && !pending_arrival && !r.pending_current_colour.enabled &&
        (!r.hdr_input || check_retained_rgb || r.native_p010)), "Packed continuity regression requires the ordinary converter");
      const UINT w=r.frames[0].width,h=r.frames[0].height;
      require(w&&h&&w<=5120u&&h<=5120u,"Invalid source extent");
      require(!r.hdr_input || (!r.queued && (check_retained_rgb || r.native_p010 || (r.selected_matte && r.selected_matte->current_outline))),
        "HDR fixture requires an explicit ordinary shutter or current-outline route");
      const UINT pixel_bytes=r.hdr_input?8u:4u;
      const auto payload=[&](const frame_t &f)->const void* {return r.hdr_input?static_cast<const void*>(f.rgba16.data()):static_cast<const void*>(f.bgra.data());};
      const auto payload_size=[&](const frame_t &f){return r.hdr_input?f.rgba16.size()*2u:f.bgra.size();};
      for(const auto &f:r.frames)require(f.width==w&&f.height==h&&payload_size(f)==std::size_t(w)*h*pixel_bytes&&f.source_file_sha256.size()==64,"Source shape/binding mismatch");
      std::filesystem::create_directories(root);created=true;
      json sources=json::array();for(std::size_t i=0;i<3;++i){auto p=root/("input_"+std::to_string(i)+(r.hdr_input?".rgba16f":".bgra8"));std::ofstream f(p,std::ios::binary);f.write(static_cast<const char*>(payload(r.frames[i])),static_cast<std::streamsize>(payload_size(r.frames[i])));f.flush();require(f.good(),"Input raw write failed");f.close();json row{{"raw",binding(p)},{"declared_timeline_us_METADATA_ONLY",r.frames[i].declared_timeline_us}};row[r.hdr_input?"declared_PFM_sha256":"declared_PNG_sha256"]=r.frames[i].source_file_sha256;sources.push_back(row);}
      require(models::prepare_tensorrt_model(SUNSHINE_ASSETS_DIR,::video::host_sbs_v2_depth_model(),config::video.adapter_name),"Resident depth preparation failed");
      require(models::prepare_ocr_tensorrt_model(SUNSHINE_ASSETS_DIR,config::video.adapter_name),"Resident OCR preparation failed");
      if(!sbs_reprojection_vs_hlsl)require(::platf::dxgi::init()==0,"Platform shader initialization failed");
      ComPtr<IDXGIFactory1> factory;hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)),"Factory creation failed");
      // Explicit fixture-machine identity, bound to the saved prior hardware diagnostic and
      // read-only DXGI census. Never choose the first NVIDIA entry or a name-only match.
      const bool explicit_adapter=r.hdr_input||r.explicit_adapter;
      const DWORD required_luid_low=explicit_adapter?r.hdr_adapter_luid_low:3780540602u;
      const LONG required_luid_high=explicit_adapter?r.hdr_adapter_luid_high:0;
      require(!explicit_adapter||((required_luid_low!=0u||required_luid_high!=0)&&r.hdr_adapter_vendor_id!=0u&&r.hdr_adapter_device_id!=0u),"Explicit probe adapter identity missing");
      const auto adapter_json=[](const DXGI_ADAPTER_DESC1 &d){
        const int n=WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,d.Description,-1,nullptr,0,nullptr,nullptr);
        require(n>0,"Adapter description UTF8 sizing failed");std::string description(static_cast<std::size_t>(n),'\0');
        require(WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,d.Description,-1,description.data(),n,nullptr,nullptr)==n,"Adapter description UTF8 conversion failed");description.resize(static_cast<std::size_t>(n-1));
        return json{{"description",description},{"vendor_id",d.VendorId},{"device_id",d.DeviceId},{"subsystem_id",d.SubSysId},{"revision",d.Revision},
          {"luid_low",d.AdapterLuid.LowPart},{"luid_high",d.AdapterLuid.HighPart},{"flags",d.Flags},
          {"dedicated_video_memory",d.DedicatedVideoMemory},{"dedicated_system_memory",d.DedicatedSystemMemory},{"shared_system_memory",d.SharedSystemMemory}};
      };
      ComPtr<IDXGIAdapter1> adapter;UINT matches=0;DXGI_ADAPTER_DESC1 adapter_desc{};
      for(UINT i=0;;++i){ComPtr<IDXGIAdapter1>a;const auto status=factory->EnumAdapters1(i,&a);if(status==DXGI_ERROR_NOT_FOUND)break;hr(status,"Adapter enumeration failed");
        DXGI_ADAPTER_DESC1 d{};hr(a->GetDesc1(&d),"Adapter desc failed");auto row=adapter_json(d);row["index"]=i;adapter_entries.push_back(row);
        if(d.AdapterLuid.LowPart==required_luid_low&&d.AdapterLuid.HighPart==required_luid_high){adapter=a;adapter_desc=d;selected_adapter=row;++matches;}}
      require(matches==1&&adapter_desc.VendorId==0x10de&&(!explicit_adapter||(adapter_desc.VendorId==r.hdr_adapter_vendor_id&&adapter_desc.DeviceId==r.hdr_adapter_device_id))&&!(adapter_desc.Flags&DXGI_ADAPTER_FLAG_SOFTWARE),"Explicit fixture NVIDIA LUID unavailable or ambiguous; no fallback");
      const auto actual_device_adapter=[&](ID3D11Device *device){
        ComPtr<IDXGIDevice> dxgi;hr(device->QueryInterface(IID_PPV_ARGS(&dxgi)),"Actual device DXGI query failed");
        ComPtr<IDXGIAdapter> base;hr(dxgi->GetAdapter(&base),"Actual device GetAdapter failed");ComPtr<IDXGIAdapter1> a;hr(base.As(&a),"Actual adapter1 query failed");
        DXGI_ADAPTER_DESC1 d{};hr(a->GetDesc1(&d),"Actual device adapter desc failed");
        require(d.AdapterLuid.LowPart==required_luid_low&&d.AdapterLuid.HighPart==required_luid_high&&d.VendorId==adapter_desc.VendorId&&d.DeviceId==adapter_desc.DeviceId,
          "Actual device adapter differs from explicit selected LUID");return adapter_json(d);
      };
      display=std::make_shared<fixture_display_t>();display->width=display->width_before_rotation=w;display->height=display->height_before_rotation=h;
      display->env_width=w;display->env_height=h;display->display_rotation=DXGI_MODE_ROTATION_IDENTITY;display->capture_format=r.hdr_input?DXGI_FORMAT_R16G16B16A16_FLOAT:DXGI_FORMAT_B8G8R8A8_UNORM;display->declared_hdr=r.hdr_input;display->next_image_id=1;
      const D3D_FEATURE_LEVEL levels[]{D3D_FEATURE_LEVEL_11_1,D3D_FEATURE_LEVEL_11_0};
      hr(D3D11CreateDevice(adapter.Get(),D3D_DRIVER_TYPE_UNKNOWN,nullptr,D3D11_CREATE_DEVICE_BGRA_SUPPORT,levels,2,D3D11_SDK_VERSION,&display->device,nullptr,&display->device_ctx),"Capture-side fixture device failed");
      capture_device_adapter=actual_device_adapter(display->device.get());
      require(converter.init(display,adapter.Get(),r.native_p010 ? pix_fmt_e::p010 : pix_fmt_e::nv12)==0,"Real converter init failed");
      converter_device_adapter=actual_device_adapter(converter.device.get());
      const context_t context{w,h,0x4356524700000001ull,r.packed_continuity};converter.probe_context=&context;
      auto settings=r.settings;settings.reshade=false; // only authored source, not an external provider
      D3D11_TEXTURE2D_DESC td {};
      td.Width = 2u * w;
      // NV12 requires even dimensions. Keep the archived source extent and its pixels intact.
      td.Height = native_nv12 ? (h + 1u) & ~1u : h;
      td.MipLevels = 1;
      td.ArraySize = 1;
      td.Format = r.native_p010 ? DXGI_FORMAT_P010 : native_nv12 ? DXGI_FORMAT_NV12 : r.hdr_input ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_B8G8R8A8_UNORM;
      td.SampleDesc.Count = 1;
      td.Usage = D3D11_USAGE_DEFAULT;
      td.BindFlags = native_nv12 ? D3D11_BIND_RENDER_TARGET : D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
      if (r.native_p010) {
        require(w % 2u == 0u && h % 2u == 0u, "Native P010 fixture requires even source dimensions");
        if (r.p010_chroma_mode != "raster") {
          require(converter.prepare_p010_chroma_tiles(), "Native P010 tile program preparation failed");
        }
        native_encoder = std::make_unique<nvenc::nvenc_d3d11_native>(converter.device.get(), r.p010_chroma_mode != "no-uav");
        ::video::config_t client {};
        client.width = td.Width;client.height = td.Height;
        client.framerate = 90;client.framerateX100 = 9000;client.encodingFramerate = 90000;
        client.bitrate = 166170;client.slicesPerFrame = 1;client.numRefFrames = 1;
        client.encoderCscMode = 2;client.videoFormat = native_codec;client.dynamicRange = 1;client.sbs_mode = ::video::SBS_AI;
        nvenc::nvenc_config encoder_config;
        encoder_config.quality_preset = 1;encoder_config.two_pass = nvenc::nvenc_two_pass::disabled;
        encoder_config.vbv_percentage_increase = 100;encoder_config.adaptive_quantization = false;
        encoder_config.hevc_unidirectional_b = false;encoder_config.split_encode = false;
        const nvenc::nvenc_colorspace_t native_colorspace {NV_ENC_VUI_COLOR_PRIMARIES_BT2020,
          NV_ENC_VUI_TRANSFER_CHARACTERISTIC_SMPTE2084,NV_ENC_VUI_MATRIX_COEFFS_BT2020_NCL,true};
        require(native_encoder->create_encoder(encoder_config, client, native_colorspace,
          NV_ENC_BUFFER_FORMAT_YUV420_10BIT, std::nullopt) && native_encoder->pipeline_depth() == 2u &&
          native_encoder->input_surface_count() == 3u, "Native P010 depth2 owner unavailable");
        target = native_encoder->get_input_texture(0u);
        require(converter.apply_colorspace({::video::colorspace_e::bt2020,true,10u}), "Native P010 colorspace setup failed");
        require(converter.init_output(target.Get(),td.Width,td.Height,::video::SBS_AI,settings,
          {},{},{},{},0,{},false,{},{},{},false,0,0,native_encoder.get()) == 0,
          "Actual native P010 converter output setup failed");
        require(converter.set_stream_gamma(::video::stream_gamma_mode_e::gamma_2_4), "Native P010 gamma2 setup failed");
      } else if (native_nv12) {
        hr(converter.device->CreateTexture2D(&td, nullptr, &target), "Native NV12 input target failed");
        require(converter.apply_colorspace({::video::colorspace_e::rec709, false, 8u}), "Native NV12 colorspace initialization failed");
        require(converter.init_output(target.Get(), td.Width, td.Height, ::video::SBS_AI, settings) == 0,
          "Real native NV12 converter initialization failed");
        const auto &output = converter.active_output_views();
        require(output.y_or_yuv && output.uv, "Native NV12 converter did not bind both plane render targets");
      } else {
        require(converter.init_rgb_output(w*2,h,::video::SBS_AI,settings)==0,"Real RGB-only initialization failed");
      }
      const auto prepare_deadline=clock_t::now()+std::chrono::seconds(60);
      while(!converter.ensure_depth_estimator()){require(clock_t::now()<prepare_deadline&&converter.host_sbs_renderer!=models::host_sbs_renderer_e::failed_flat,"Estimator preparation unavailable");std::this_thread::sleep_for(std::chrono::milliseconds(1));}
      est=converter.depth_estimator.get();require(est&&est->is_valid(),"Real estimator unavailable");
      if(r.temporal_source_bridge){std::string setup_error;require(converter.configure_temporal_source_bridge_for_instance(*r.temporal_source_bridge,setup_error),("Temporal source bridge pre-input setup: "+setup_error).c_str());}
      if(r.selected_matte){std::string setup_error;require(converter.configure_selected_matte_for_instance(*r.selected_matte,setup_error),("Selected matte pre-input setup: "+setup_error).c_str());}
      if(r.queued)require(converter.initialize_queued_renderer_for_instance(),"Per-instance queued resources unavailable");
      models::joined_completion_gate_benchmark::configuration_t gate;gate.enabled=true;
      gate.placement=r.packed_continuity?models::joined_completion_gate_benchmark::placement_e::after_interop_unmaps:models::joined_completion_gate_benchmark::placement_e::before_interop_unmaps;
      gate.target_frame_id=cached_route?4u:3u;gate.hold_after_callback_enter_ms=r.packed_continuity&&!borrowed_recycle?350u:120u;gate.callback_safety_timeout_ms=500;
      require(est->configure_joined_completion_gate_for_benchmark(gate),"Pre-input gate configuration failed");
      if (!native_nv12 && !r.native_p010) {
        hr(converter.device->CreateTexture2D(&td,nullptr,&target),"RGB target failed");hr(converter.device->CreateRenderTargetView(target.Get(),nullptr,&target_rtv),"RGB target RTV failed");
      }
      for(std::size_t i=0;i<3;++i){images[i]=display->alloc_img();require(images[i]&&display->complete_img(images[i].get(),false)==0,"Actual shared image allocation failed");auto &img=static_cast<img_d3d_t&>(*images[i]);detail::keyed_mutex_lock_t lock {img.capture_mutex.get()};require(lock.lock() == S_OK,"Capture keyed mutex failed");display->device_ctx->UpdateSubresource(img.capture_texture.get(),0,nullptr,payload(r.frames[i]),w*pixel_bytes,0);display->device_ctx->Flush();img.blank=false;}
      // Open every shared image before input1 without establishing frame or publication ownership.
      for(const auto &image:images){auto &img=static_cast<img_d3d_t&>(*image);auto [it,inserted]=converter.img_ctx_map.try_emplace(img.id);(void)inserted;require(converter.initialize_image_context(img,it->second)==0,"Shared image preparation failed");}
      if (!context.borrow_matched_capture) {
        D3D11_TEXTURE2D_DESC private_desc{};static_cast<img_d3d_t&>(*images[0]).capture_texture->GetDesc(&private_desc);
        private_desc.Usage=D3D11_USAGE_DEFAULT;private_desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;private_desc.MiscFlags=0;private_desc.CPUAccessFlags=0;
        for(auto &slot:converter.matched_frame_slots){hr(converter.device->CreateTexture2D(&private_desc,nullptr,&slot.texture),"Private slot preparation failed");hr(converter.device->CreateShaderResourceView(slot.texture.get(),nullptr,&slot.srv),"Private slot SRV failed");}
      }
      records.ctx=converter.device_ctx.get();converter.probe_observer={&records,&recorder_t::record};
      const auto stamp=[&](std::size_t i){const auto t=clock_t::now();images[i]->frame_timestamp=t;images[i]->content_timestamp=t;};
      const auto convert = [&](std::size_t i) {
        const auto deadline = clock_t::now() - std::chrono::milliseconds(1);
        const bool converted = (native_nv12 || r.native_p010) ? converter.convert(*images[i], deadline) == 0 :
                                             converter.convert_rgb(*images[i], target.Get(), target_rtv.Get(), r.hdr_input, deadline) ==
                                               d3d_base_encode_device::rgb_conversion_e::converted;
        require(converted, "Actual converter failed");
        require(!records.overflow && !records.capture_fault, "Passive observer unavailable");
      };
      const auto wait_ready=[&](){const auto until=clock_t::now()+std::chrono::seconds(30);while(!est->can_accept_frame()){require(!est->has_terminal_failure()&&clock_t::now()<until,"Real root readiness timeout");std::this_thread::sleep_for(std::chrono::milliseconds(1));}};
      ComPtr<ID3D11Device> shutter_consumer;
      ComPtr<ID3D11DeviceContext> shutter_context;
      local_shutter_frame_ring_t shutter_ring;
      using rgb_result_e = d3d_base_encode_device::rgb_conversion_e;
      using rgb_destination_e = d3d_base_encode_device::rgb_destination_e;
      const auto acquire_shutter_pair = [&](ID3D11Texture2D *previous) {
        const auto limit = clock_t::now() + std::chrono::seconds(2);
        while (!shutter_ring.failed() && clock_t::now() < limit) {
          auto *pair = shutter_ring.acquire_pair();
          if (pair && pair != previous) {
            return pair;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        throw std::runtime_error("The consumer did not acquire the complete shutter pair");
      };
      const auto convert_shutter = [&](std::size_t i, const local_shutter_frame_ring_t::slot_t &slot, rgb_destination_e destination) {
        const auto outcome = converter.convert_rgb(*images[i], slot.producer_texture.Get(), slot.producer_target.Get(),
          r.hdr_input, clock_t::now() - std::chrono::milliseconds(1), destination);
        require(!records.overflow && !records.capture_fault, "Passive observer unavailable");
        return outcome;
      };
      const auto changed_shutter_destination = [&]() {
        return copy_reference ? rgb_destination_e::backbuffer : shutter_ring.rgb_destination();
      };
      ID3D11Texture2D *initial_shutter_pair = nullptr;
      stage_t initial_rgb_output;
      stamp(0);
      if (check_retained_rgb) {
        hr(D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
          levels, 2, D3D11_SDK_VERSION, &shutter_consumer, nullptr, &shutter_context), "Shutter consumer device failed");
        require(shutter_ring.init(converter.device.get(), shutter_consumer.Get(), td.Width, td.Height, td.Format),
          "Actual shutter packet allocation failed");
        auto *slot = shutter_ring.acquire_producer();
        require(slot && !shutter_ring.has_published_pair() && !converter.sbs_intermediate_texture,
          "Initial shutter slot or lazy scratch precondition unavailable");
        require(convert_shutter(0, *slot, changed_shutter_destination()) == rgb_result_e::converted &&
          shutter_ring.finish_producer(*slot, true, converter.device_ctx.get()), "Initial shutter frame was not published");
        require(static_cast<bool>(converter.sbs_intermediate_texture) == copy_reference,
          "Initial shutter frame used the wrong packed storage");
        initial_shutter_pair = acquire_shutter_pair(nullptr);
        initial_rgb_output.prepare(shutter_consumer.Get(), initial_shutter_pair);
        require(initial_rgb_output.copy(shutter_context.Get(), initial_shutter_pair), "Initial shutter packet copy failed");
        shutter_context->Flush();
      } else {
        convert(0);
      }
      wait_ready();
      convert(0); // exact same img/source/content identity on retry
      const auto *a=records.last_draw();require(a&&a->publication_current&&!a->queued&&a->publication.frame_id==1&&converter.sbs_frame_sequence==2,"A/ACK protocol unavailable; no invented ordinal");
      require(std::none_of(converter.matched_frame_slots.begin(),converter.matched_frame_slots.end(),[](const auto &s){return s.pending;}),"A retry unexpectedly enqueued; target3 unavailable");
      if(!r.packed_continuity) {
        // convert() has released its encoder-side keyed lease. Reacquire that SAME
        // prepared A context before the initial diagnostic reads its shared SRV.
        const auto a_context=converter.img_ctx_map.find(static_cast<img_d3d_t&>(*images[0]).id);
        require(a_context!=converter.img_ctx_map.end()&&a_context->second.encoder_mutex&&
          a_context->second.encoder_input_res.get()==a->current_input.Get(),"Prepared A shared context identity unavailable");
        detail::keyed_mutex_lock_t a_snapshot_lease {a_context->second.encoder_mutex.get()};
        require(a_snapshot_lease.lock() == S_OK,"Initial A shared diagnostic keyed lease failed");
        records.prepare(converter.device.get(),*a,target.Get());records.copy_target(0,target.Get());
        converter.device_ctx->Flush(); // submit every shared read BEFORE ReleaseSync(0)
      }
      const auto a_publication=a->publication;
      if (cached_route) {
        require(!a->flat && !a->repeat && !a->region.is_video_region() && images[0]->content_timestamp, "A must establish genuine full-source warped color and content clock");
        const auto original = *a;
        const auto owner = est;
        std::array<stage_t, 6u> output;
        for (std::size_t i = 0u; i < 4u; ++i) output[i].prepare(converter.device.get(), target.Get());
        const auto base = resource(original.views[1].Get());require(base != nullptr, "A actual Base binding missing");
        output[4].prepare(converter.device.get(), base.Get());output[5].prepare(converter.device.get(), base.Get());
        require(output[0].copy(converter.device_ctx.get(), target.Get()) && output[4].copy(converter.device_ctx.get(), base.Get()), "A reference staging failed");
        auto &fresh_image = static_cast<img_d3d_t &>(*images[1]);
        const auto &fresh_context = converter.img_ctx_map.at(fresh_image.id);
        {
          detail::keyed_mutex_lock_t lease {fresh_image.capture_mutex.get()};require(lease.lock() == S_OK, "Fresh capture upload key failed");
          display->device_ctx->UpdateSubresource(fresh_image.capture_texture.get(), 0u, nullptr, payload(r.frames[0]), w * pixel_bytes, 0u);
          display->device_ctx->Flush();
        }
        stamp(1);images[1]->content_timestamp = images[0]->content_timestamp;
        require(images[1]->frame_timestamp != images[0]->frame_timestamp, "Fresh capture must have a distinct source clock");
        convert(1);
        const auto *observed_cached = records.last_draw();require(observed_cached != nullptr, "Cached current draw absent");
        const auto cached = *observed_cached;
        require(cached.conversion == 3u && cached.cached_current_colour && !cached.flat && !cached.repeat && !cached.queued &&
          cached.publication_current && cached.publication == a_publication && cached.model_frame == a_publication.frame_id &&
          cached.actual_draw_pixel_source == 3u && cached.source_stamp_ticks == images[1]->frame_timestamp->time_since_epoch().count() &&
          cached.content_stamp_ticks == images[0]->content_timestamp->time_since_epoch().count() &&
          cached.views[0].Get() == fresh_context.encoder_input_res.get() && cached.current_input.Get() == cached.views[0].Get() &&
          cached.views[0].Get() != original.views[0].Get() && cached.views[1].Get() == original.views[1].Get() &&
          std::count_if(records.events.begin(), records.events.begin() + records.count, [](const auto &e) { return e.kind == event_e::admission; }) == 1,
          "Exact full-source proof did not draw new capture color using A's genuine depth publication");
        require(output[1].copy(converter.device_ctx.get(), target.Get()) && output[5].copy(converter.device_ctx.get(), base.Get()), "Cached output staging failed");
        ComPtr<ID3D11Texture2D> flat_reference;ComPtr<ID3D11RenderTargetView> flat_reference_rtv;
        hr(converter.device->CreateTexture2D(&td, nullptr, &flat_reference), "Route flat reference target failed");
        hr(converter.device->CreateRenderTargetView(flat_reference.Get(), nullptr, &flat_reference_rtv), "Route flat reference RTV failed");
        const context_t changed_context {w, h, context.instance + 1u, context.borrow_matched_capture};
        require(r.hdr_input ? r.frames[0].rgba16 != r.frames[2].rgba16 : r.frames[0].bgra != r.frames[2].bgra, "Changed-route current pixels must differ from A");
        converter.probe_context = &changed_context;stamp(2);convert(2);
        require(converter.depth_estimator.get() == owner, "Route transition replaced the warmed estimator");
        const auto changed_generation = converter.subtitle_context.generation();
        const auto *observed_flat = records.last_draw();require(observed_flat != nullptr, "Changed-context draw absent");
        const auto flat = *observed_flat;
        auto &current_context = converter.img_ctx_map.at(static_cast<img_d3d_t &>(*images[2]).id);
        require(flat.conversion == 4u && flat.flat && !flat.repeat && !flat.queued && !flat.cached_current_colour &&
          flat.views[0].Get() == current_context.encoder_input_res.get() && flat.views[0].Get() == flat.current_input.Get() &&
          flat.current_source_stamp_ticks == images[2]->frame_timestamp->time_since_epoch().count() && flat.watermark_after >= 4u &&
          changed_generation != cached.context_generation && !converter.latest_v2_lineage.authenticated(),
          "Changed synthetic route reused old protected pixels or analysis");
        require(std::any_of(records.events.begin(), records.events.begin() + records.count, [](const auto &e) { return e.kind == event_e::admission && e.conversion == 4u && e.model_frame == 4u && e.force_infer && !e.queued && e.optional_work == static_cast<std::uint32_t>(models::depth_optional_work_mode_e::ordinary); }), "Changed route did not admit its own ordinary force-infer root4");
        const auto entered_until = clock_t::now() + std::chrono::seconds(2);
        std::optional<models::joined_completion_gate_benchmark::snapshot_t> held;
        do { held = est->joined_completion_gate_status_for_benchmark();require(held && !held->released && clock_t::now() < entered_until, "Changed-route root gate unavailable");if (!held->callback_entered) std::this_thread::yield(); } while (!held->callback_entered);
        const auto color = r.hdr_input ? models::input_color_space::scrgb_hdr : models::input_color_space::srgb;
        {
          detail::keyed_mutex_lock_t lease {current_context.encoder_mutex.get()};require(lease.lock() == S_OK, "Changed-route source key failed");
          require(!est->try_finish_pending_depth_nonblocking(color, false).ready, "Changed-route root unexpectedly completed");
          held = est->joined_completion_gate_status_for_benchmark();
          require(held && held->configuration.target_frame_id == 4u && held->last_actual_event_query == CUDA_ERROR_NOT_READY && !held->released, "No literal pending root4 evidence");
          host_sbs_v2_draw_command_t draw {.render_targets={flat_reference_rtv.Get(),nullptr}, .vertex_shader=converter.sbs_reprojection_vs.get(),
            .pixel_shader=converter.sbs_flat_identity_ps.get(), .viewport=converter.sbs_viewport, .sampler=converter.sampler_linear.get(),
            .shader_resources={flat.current_input.Get(),nullptr,nullptr,nullptr,nullptr,nullptr,nullptr}, .geometry_constants=flat.buffers[0].Get()};
          require(record_host_sbs_v2_draw(converter.device_ctx.get(), draw) && output[2].copy(converter.device_ctx.get(), target.Get()) &&
            output[3].copy(converter.device_ctx.get(), flat_reference.Get()), "Changed-route flat reference recording failed");
          converter.device_ctx->Flush();
        }
        wait_ready();const auto released = est->joined_completion_gate_status_for_benchmark();
        require(released && released->released && released->callback_exited && !released->cancelled && !released->callback_timed_out && !released->callback_entry_timed_out, "Changed-route root did not release normally");
        convert(2);const auto *recovered = records.last_draw();
        require(recovered && recovered->publication_current && recovered->publication.frame_id == 4u && !recovered->flat &&
          !recovered->repeat && !recovered->cached_current_colour && recovered->pending_slots == 0u && recovered->context_generation == changed_generation,
          "Changed-route root did not ACK under its own new context");
        json artifacts = json::array();constexpr std::array<const char *, 6u> roles {"matched-A", "cached-current-A", "changed-route-flat", "canonical-current-flat", "matched-Base", "cached-Base"};
        for (std::size_t i = 0u; i < output.size(); ++i) { output[i].read(converter.device_ctx.get());const auto path = root / (std::string(roles[i]) + (i >= 4u ? ".f32" : r.hdr_input ? ".rgba16f" : ".bgra8"));
          std::ofstream f(path, std::ios::binary);f.write(output[i].bytes.data(), output[i].bytes.size());f.close();require(f.good(), "Cached-route artifact write failed");auto artifact = binding(path);artifact["role"] = roles[i];artifacts.push_back(artifact); }
        require(output[0].bytes == output[1].bytes && output[4].bytes == output[5].bytes && output[2].bytes == output[3].bytes,
          "Cached depth/current color or changed-route flat output differs from its literal reference");
        write_json(root / "result.json", {{"schema",1}, {"status","PASS_FUNCTIONAL_CACHED_CURRENT_ROUTE"}, {"scope","actual converter; synthetic route transition; full-source exact content-clock proof; no live eligibility or latency claim"},
          {"source_inputs",sources}, {"fresh_capture_payload_parent_index",0}, {"overwritten_archived_input_index",1}, {"cached_depth_publication",pub(cached.publication)},
          {"A",event_json(original)}, {"cached_current",event_json(cached)}, {"changed_route_flat",event_json(flat)}, {"changed_route_recovered",event_json(*recovered)},
          {"context_instances",{context.instance,changed_context.instance}}, {"context_generations",{cached.context_generation,changed_generation}}, {"cached_event_observation_us_diagnostic",cached.observation_us},
          {"cached_model_observation_us",cached.publication.observation_timestamp_us}, {"gate_held",gate_json(held)}, {"gate_released",gate_json(released)}, {"outputs",artifacts}});
        converter.probe_context = nullptr;converter.probe_observer = {};return 0;
      }
      if(r.packed_continuity) {
        const auto a_pixel_source_frame=a->actual_draw_pixel_source;
        const auto a_source_stamp=a->source_stamp_ticks;
        require(a_pixel_source_frame!=0u&&a_source_stamp==a->current_source_stamp_ticks,
          "A actual RGB draw/source identity unavailable");
        // Only source arrival and the real joined-event gate are controlled. Every output below
        // comes from the actual converter observer and ordered copies of its output target.
        require(r.hdr_input ? r.frames[0].rgba16 != r.frames[1].rgba16 : r.frames[0].bgra != r.frames[1].bgra,
          "A and B must have distinct authenticated source bytes");
        const auto no_private_matched_texture = [&]() {
          return std::none_of(converter.matched_frame_slots.begin(), converter.matched_frame_slots.end(),
            [](const auto &slot) { return slot.texture && !slot.capture_image; });
        };
        require(a->views[0].Get() == a->current_input.Get() && images[0].use_count() == 1 &&
          no_private_matched_texture(), "Completed A did not render from and release its shared capture image");
        auto &b_image = static_cast<img_d3d_t &>(*images[1]);
        auto &b_context = converter.img_ctx_map.at(b_image.id);
        const auto free_pool_image = [&]() {
          return std::find_if(images.begin(), images.end(), [](const auto &image) { return image.use_count() == 1; });
        };
        const auto check_pending_b_pin = [&]() {
          const auto slot = std::find_if(converter.matched_frame_slots.begin(), converter.matched_frame_slots.end(),
            [](const auto &candidate) { return candidate.pending && candidate.frame_id == 3u; });
          require(slot != converter.matched_frame_slots.end() &&
            slot->capture_image.get() == images[1].get() &&
            slot->texture.get() == b_context.encoder_texture.get() &&
            slot->srv.get() == b_context.encoder_input_res.get() && images[1].use_count() > 1 &&
            no_private_matched_texture(), "Pending B did not pin the exact shared capture source");
          const auto reserve_a = images[0];
          require(free_pool_image() == images.begin() + 2,
            "Three-image pool did not skip the pinned B source for C");
        };
        const auto check_released_b_pin = [&]() {
          const auto reserve_a = images[0];
          const auto reserve_c = images[2];
          require(free_pool_image() == images.begin() + 1 &&
            std::none_of(converter.matched_frame_slots.begin(), converter.matched_frame_slots.end(),
              [&](const auto &slot) { return slot.capture_image.get() == images[1].get(); }) &&
            no_private_matched_texture(), "Completed B did not release its exact capture image for recycle");
        };
        const auto wait_for_b_gate_entry = [&]() {
          const auto entered_deadline=clock_t::now()+std::chrono::milliseconds(100);
          for(;;) {
            const auto status=est->joined_completion_gate_status_for_benchmark();
            require(status&&!status->released&&clock_t::now()<entered_deadline,"No bounded held B callback window");
            if(status->callback_entered) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
          }
        };
        const auto real_pending = [&]() {
          detail::keyed_mutex_lock_t lease {b_context.encoder_mutex.get()};
          require(lease.lock(0, 100) == S_OK, "Diagnostic B completion query could not acquire its source key");
          const auto color = r.hdr_input ? models::input_color_space::scrgb_hdr : models::input_color_space::srgb;
          const auto pending=est->try_finish_pending_depth_nonblocking(color,false);
          const auto held=est->joined_completion_gate_status_for_benchmark();
          require(!pending.ready&&held&&held->queued&&held->callback_entered&&!held->released&&
            !held->cancelled&&!held->callback_timed_out&&!held->callback_entry_timed_out&&
            held->configuration.placement==models::joined_completion_gate_benchmark::placement_e::after_interop_unmaps&&
            held->configuration.hold_after_callback_enter_ms==gate.hold_after_callback_enter_ms&&
            held->configuration.callback_safety_timeout_ms==gate.callback_safety_timeout_ms&&
            held->last_actual_event_query==CUDA_ERROR_NOT_READY&&held->actual_not_ready_queries>0u,
            "B is not an actually queried held root; no manufactured pending result");
          require(std::count_if(converter.matched_frame_slots.begin(),converter.matched_frame_slots.end(),
            [](const auto &s){return s.pending&&s.frame_id==3u;})==1,"Held B lost its sole pending slot");
          check_pending_b_pin();
          return *held;
        };
        const auto last_outcome=[&](std::uint64_t conversion) {
          for(std::size_t i=records.count;i>0;--i) {
            const auto &e=records.events[i-1u];
            if(e.conversion==conversion&&(e.kind==event_e::packed_repeat||e.kind==event_e::convert_end))return e;
          }
          throw std::runtime_error("Actual converter outcome missing");
        };
        stage_t recycled_b_source;
        if (borrowed_recycle) {
          records.borrowed_draw_sources[0].prepare(converter.device.get(), b_context.encoder_texture.get());
          records.borrowed_draw_sources[1].prepare(converter.device.get(), b_context.encoder_texture.get());
          recycled_b_source.prepare(converter.device.get(), b_context.encoder_texture.get());
        }
        std::array<stage_t, 6u> outputs;
        const auto output_count = native_nv12 ? outputs.size() : outputs.size() - 1u;
        for (std::size_t i = 0; i < output_count; ++i) {
          if ((borrowed_recycle && i != 0u && i != 3u) || (check_retained_rgb && i == 3u)) {
            continue;
          }
          outputs[i].prepare(converter.device.get(), target.Get());
        }
        const auto snapshot_target = [&](std::size_t index, const char *message) {
          if (native_nv12 || r.native_p010) {
            converter.device_ctx->OMSetRenderTargets(0, nullptr, nullptr);
          }
          auto *actual = r.native_p010 ? native_encoder->get_input_texture(converter.active_output_surface) : target.Get();
          require(outputs[index].copy(converter.device_ctx.get(), actual), message);
        };
        json native_input_proof = json::object();
        const auto record_native_input = [&](const char *phase, const event_t &output, bool expect_unchanged) {
          require(output.conversion == converter.sbs_frame_sequence &&
                  (output.kind == event_e::packed_repeat || output.kind == event_e::convert_end),
            "Native NV12 input proof does not name the latest successful conversion output");
          const bool unchanged = !output.native_planes_written;
          const bool kept = converter.conversion_kept_input();
          native_input_proof[phase] = {{"encoder_input_pixels_unchanged", unchanged}, {"conversion_kept_input", kept},
            {"conversion", output.conversion}, {"native_planes_written", output.native_planes_written}};
          require(unchanged == expect_unchanged && !kept, "Native NV12 pixel identity disagrees with the actual converter phase");
        };
        ComPtr<ID3D11Texture2D> flat_reference;
        ComPtr<ID3D11RenderTargetView> flat_reference_rtv;
        if (native_nv12 || r.native_p010) {
          record_native_input("matched_A", last_outcome(2u), false);
        } else {
          hr(converter.device->CreateTexture2D(&td,nullptr,&flat_reference),"Independent flat reference target failed");
          hr(converter.device->CreateRenderTargetView(flat_reference.Get(),nullptr,&flat_reference_rtv),"Independent flat reference RTV failed");
        }
        const auto a_content_timestamp = converter.rendered_content_timestamp();
        snapshot_target(0, "Actual A target copy failed");
        json native_p010_proof = json::object();
        unsigned native_a_surface = 0;
        if (r.native_p010) {
          const auto a_output = last_outcome(2u);
          const bool expect_tiles = r.p010_chroma_mode == "default";
          require(a_output.native_planes_written && a_output.p010_chroma_tiled == expect_tiles,
            "Matched A used the wrong native P010 conversion program");
          native_a_surface = converter.active_output_surface;
          outputs[0].read(converter.device_ctx.get()); // Before mapping A; outside held B interval.
          D3D11_TEXTURE2D_DESC desc {};
          native_encoder->get_input_texture(native_a_surface)->GetDesc(&desc);
          require(((desc.BindFlags & D3D11_BIND_UNORDERED_ACCESS) != 0u) == (r.p010_chroma_mode != "no-uav"),
            "Native P010 fixture allocation did not exercise the requested UAV policy");
          require(native_encoder->submit_frame(1u,true) && native_encoder->frames_in_flight() == 1u,
            "Actual matched A native submission failed");
          native_p010_proof = {{"mode",r.p010_chroma_mode},{"A_surface",native_a_surface},
            {"A_tiled",a_output.p010_chroma_tiled},{"A_bind_flags",desc.BindFlags},
            {"gamma_mode",static_cast<unsigned>(converter.stream_gamma_mode())},
            {"gamma_white_nits",converter.stream_gamma_white_nits()}};
        }
        ID3D11Texture2D *matched_a_shutter_pair = nullptr;
        stage_t unused_rgb_target, retained_rgb_pair;
        if (check_retained_rgb) {
          auto *slot = shutter_ring.acquire_producer();
          require(slot, "Shutter slot for matched A unavailable");
          converter.device_ctx->CopyResource(slot->producer_texture.Get(), target.Get());
          require(shutter_ring.finish_producer(*slot, true, converter.device_ctx.get()), "Matched A shutter publication failed");
          matched_a_shutter_pair = acquire_shutter_pair(initial_shutter_pair);
          retained_rgb_pair.prepare(shutter_consumer.Get(), matched_a_shutter_pair);
          unused_rgb_target.prepare(converter.device.get(), target.Get());
        }
        converter.device_ctx->Flush();
        const auto idle_begin=clock_t::now();
        std::this_thread::sleep_until(idle_begin+std::chrono::milliseconds(300));
        const auto idle_end=clock_t::now();
        stamp(1);
        const auto b_begin=clock_t::now();
        if (check_retained_rgb) {
          auto *slot = shutter_ring.acquire_producer();
          require(slot != nullptr, "Shutter producer could not reserve a free slot for B");
          const float canary[4] {};
          converter.device_ctx->ClearRenderTargetView(slot->producer_target.Get(), canary);
          require(convert_shutter(1, *slot, shutter_ring.rgb_destination()) == rgb_result_e::retained,
            "The retained shutter pair unexpectedly rendered to the free slot");
          require(unused_rgb_target.copy(converter.device_ctx.get(), slot->producer_texture.Get()),
            "Copy of the unused shutter slot failed");
          require(shutter_ring.finish_producer(*slot, false, converter.device_ctx.get()) &&
                    shutter_ring.acquire_pair() == matched_a_shutter_pair &&
                    retained_rgb_pair.copy(shutter_context.Get(), matched_a_shutter_pair),
            "The consumer did not retain the original shutter packet");
          shutter_context->Flush();
        } else {
          convert(1);
        }
        const auto before=last_outcome(3u);
        require(!r.native_p010 || (!before.native_planes_written && !before.p010_chroma_tiled &&
          native_encoder->frames_in_flight() == 1u && converter.active_output_surface == native_a_surface),
          "Held B changed the mapped native A output instead of retaining it");
        const json first_checks{{"repeat",before.repeat},{"not_flat",!before.flat},{"not_queued",!before.queued},
          {"packed_valid",before.packed_valid},{"route_matches",before.route_matches},{"changed_source",!before.source_matches},
          {"actual_A_pixel_source_matches",before.pixel_source_frame==a_pixel_source_frame},
          {"actual_A_source_stamp_matches",before.pixel_source_stamp_ticks==a_source_stamp},
          {"raw_A_age_exceeds_cap",before.packed_age_us>250000},
          {"literal_B_source_matches",before.current_source_stamp_ticks==images[1]->frame_timestamp->time_since_epoch().count()}};
        result["first_changed_diagnostics"]={{"A_model_publication",pub(a_publication)},
          {"A_actual_RGB_pixel_source",a_pixel_source_frame},{"A_actual_source_stamp_ticks",a_source_stamp},
          {"actual_before_cap",event_json(before)},{"checks",first_checks}};
        const auto first_error="First changed delivery after static A did not actually repeat packed A: "+first_checks.dump();
        require(std::all_of(first_checks.begin(),first_checks.end(),[](const auto &check){return check.template get<bool>();}),
          first_error.c_str());
        require(clock_t::now()-b_begin<std::chrono::milliseconds(250),"Before-cap observation exceeded the live continuity bound");
        const auto b_admission=std::find_if(records.events.begin(),records.events.begin()+records.count,
          [](const auto &e){return e.kind==event_e::admission&&e.conversion==3u;});
        require(b_admission!=records.events.begin()+records.count&&b_admission->model_frame==3u&&
          b_admission->force_infer&&!b_admission->gpu_undecided&&
          b_admission->optional_work==static_cast<std::uint32_t>(models::depth_optional_work_mode_e::ordinary),
          "B did not take the actual ordinary force-infer admission branch");
        if (native_nv12) {
          record_native_input("packed_A_before_cap", before, true);
          require(a_content_timestamp && converter.rendered_content_timestamp() == a_content_timestamp,
            "Copy-free packed repeat changed its rendered content timestamp");
        }
        if (!borrowed_recycle) {
          snapshot_target(1, "Actual before-cap packed target copy failed");
        }
        converter.device_ctx->Flush();
        wait_for_b_gate_entry();
        const auto held_before=real_pending();
        if (borrowed_recycle) {
          auto &c_image = static_cast<img_d3d_t &>(*images[2]);
          const auto &c_context = converter.img_ctx_map.at(c_image.id);
          const auto upload_a_pixels = [&](img_d3d_t &image) {
            detail::keyed_mutex_lock_t lease {image.capture_mutex.get()};
            require(lease.lock(0, 100) == S_OK, "Reusable capture image key unavailable");
            display->device_ctx->UpdateSubresource(image.capture_texture.get(), 0, nullptr, payload(r.frames[0]), w * pixel_bytes, 0);
            display->device_ctx->Flush();
          };
          // Frame 3 of the Continuity recipe can be byte-identical to B. Reuse archived A
          // bytes on the available third image C so recycling is observable and reproducible.
          upload_a_pixels(c_image);
          stamp(2);
          wait_ready(); // Query-only preflight. The converter still owns the sole B completion.
          const auto released = est->joined_completion_gate_status_for_benchmark();
          require(released && released->released && released->callback_exited && !released->cancelled &&
            !released->callback_timed_out && !released->callback_entry_timed_out,
            "B did not become ready after its genuine held root");
          const auto timeout_conversion = converter.sbs_frame_sequence + 1u;
          {
            detail::keyed_mutex_lock_t contention {b_image.capture_mutex.get()};
            require(contention.lock(0, 100) == S_OK, "Could not induce prior-image key contention");
            convert(2);
          }
          const auto key_timeout = last_outcome(timeout_conversion);
          require(!r.native_p010 || (!key_timeout.native_planes_written && !key_timeout.p010_chroma_tiled &&
            native_encoder->frames_in_flight() == 1u && converter.active_output_surface == native_a_surface),
            "Unavailable B source changed the mapped native A output");
          const auto still_unconsumed = est->joined_completion_gate_status_for_benchmark();
          require(still_unconsumed && still_unconsumed->actual_event_queries == released->actual_event_queries + 1 &&
            still_unconsumed->last_actual_event_query == CUDA_SUCCESS &&
            key_timeout.repeat && !key_timeout.flat && key_timeout.watermark_after == key_timeout.watermark_before &&
            key_timeout.current_input.Get() == c_context.encoder_input_res.get(),
            "Unavailable B key did not leave the ready completion pending and repeat packed A");
          check_pending_b_pin();

          const auto recovery_conversion = converter.sbs_frame_sequence + 1u;
          records.borrowed_draw_conversion = recovery_conversion;
          // Current C retains its capture stamp. The ready completion still owns B.
          if (check_retained_rgb) {
            auto *slot = shutter_ring.acquire_producer();
            require(slot && convert_shutter(2, *slot, changed_shutter_destination()) == rgb_result_e::converted &&
                      shutter_ring.finish_producer(*slot, true, converter.device_ctx.get()),
              "The joined B completion did not produce a shutter pair");
          } else {
            convert(2);
          }
          records.borrowed_draw_conversion = 0;
          const auto recovered = last_outcome(recovery_conversion);
          require(!check_retained_rgb || recovered.rgb_rendered_directly == !copy_reference,
            "The qualified matched warp used the wrong packed storage");
          const auto *draw = records.last_draw();
          require(draw && draw->conversion == recovery_conversion && draw->publication_current && !draw->flat && !draw->queued &&
            draw->publication.frame_id == 3u && draw->actual_draw_pixel_source == 3u &&
            draw->source_stamp_ticks == before.current_source_stamp_ticks &&
            draw->current_source_stamp_ticks == images[2]->frame_timestamp->time_since_epoch().count() &&
            draw->current_source_stamp_ticks != draw->source_stamp_ticks &&
            draw->views[0].Get() == b_context.encoder_input_res.get() &&
            draw->current_input.Get() == c_context.encoder_input_res.get() &&
            recovered.publication_current && !recovered.repeat && recovered.query_observed &&
            recovered.literal_last_query_result == CUDA_SUCCESS && recovered.literal_ready_count > 0u,
            "Late C conversion did not render the genuine B completion from its separately keyed source");
          check_released_b_pin();
          if (check_retained_rgb) {
            auto *pair = acquire_shutter_pair(matched_a_shutter_pair);
            outputs[3].prepare(shutter_consumer.Get(), pair);
            require(outputs[3].copy(shutter_context.Get(), pair), "Direct shutter pair consumer readback failed");
          } else {
            snapshot_target(3, "Actual late B target copy failed");
          }

          if (r.native_p010) {
            const auto b_surface = converter.active_output_surface;
            require(recovered.native_planes_written && recovered.p010_chroma_tiled == (r.p010_chroma_mode == "default") &&
              b_surface != native_a_surface && native_encoder->frames_in_flight() == 1u &&
              !native_encoder->input_waits_for_retirement(),
              "B did not convert into a distinct free native P010 owner while A remained mapped");
            D3D11_TEXTURE2D_DESC desc {};
            native_encoder->get_input_texture(b_surface)->GetDesc(&desc);
            require(((desc.BindFlags & D3D11_BIND_UNORDERED_ACCESS) != 0u) == (r.p010_chroma_mode != "no-uav"),
              "B native P010 allocation differs from the requested UAV policy");
            require(native_encoder->submit_frame(2u,false) && native_encoder->frames_in_flight() == 2u,
              "Both actual native pictures were not accepted before retrieval");
            auto packet_a = native_encoder->retrieve_frame();
            auto packet_b = native_encoder->retrieve_frame();
            require(!packet_a.data.empty() && !packet_b.data.empty() && packet_a.frame_index == 1u &&
              packet_b.frame_index == 2u && packet_a.idr && native_encoder->frames_in_flight() == 0u,
              "Native HEVC/AV1 packets did not retire the exact A/B owners in FIFO order");
            const auto stream_path = root / (native_codec == 1 ? "native.hevc" : "native.obu");
            std::ofstream stream(stream_path,std::ios::binary);
            stream.write(reinterpret_cast<const char *>(packet_a.data.data()),static_cast<std::streamsize>(packet_a.data.size()));
            stream.write(reinterpret_cast<const char *>(packet_b.data.data()),static_cast<std::streamsize>(packet_b.data.size()));
            stream.flush();require(stream.good(),"Native P010 bitstream write failed");stream.close();
            native_p010_proof["B_surface"] = b_surface;
            native_p010_proof["B_tiled"] = recovered.p010_chroma_tiled;
            native_p010_proof["B_bind_flags"] = desc.BindFlags;
            native_p010_proof["submitted_before_retrieval"] = 2u;
            native_p010_proof["packets"] = json::array({
              {{"frame_index",packet_a.frame_index},{"idr",packet_a.idr},{"bytes",packet_a.data.size()}},
              {{"frame_index",packet_b.frame_index},{"idr",packet_b.idr},{"bytes",packet_b.data.size()}}});
            native_p010_proof["stream"] = binding(stream_path);
            native_p010_proof["independent_decode_qualified"] = false;
          }

          // Only after the actual matched draw and its keyed diagnostic copies are recorded can
          // the capture pool recycle B. Mutate that same shared surface before mapping any data.
          upload_a_pixels(b_image);
          stamp(1);
          {
            detail::keyed_mutex_lock_t lease {b_context.encoder_mutex.get()};
            require(lease.lock(0, 100) == S_OK &&
              recycled_b_source.copy(converter.device_ctx.get(), b_context.encoder_texture.get()),
              "Recycled B shared source readback unavailable");
            converter.device_ctx->Flush();
          }
          for (auto &source : records.borrowed_draw_sources) source.read(converter.device_ctx.get());
          recycled_b_source.read(converter.device_ctx.get());
          outputs[3].read(check_retained_rgb ? shutter_context.Get() : converter.device_ctx.get());
          if (check_retained_rgb) {
            unused_rgb_target.read(converter.device_ctx.get());
            retained_rgb_pair.read(shutter_context.Get());
            outputs[0].read(converter.device_ctx.get());
            initial_rgb_output.read(shutter_context.Get());
            require(!unused_rgb_target.bytes.empty() &&
                      std::all_of(unused_rgb_target.bytes.begin(), unused_rgb_target.bytes.end(),
                        [](char byte) { return byte == 0; }),
              "Retaining a shutter pair modified the unused RGB target");
            require(!outputs[0].bytes.empty() && retained_rgb_pair.bytes == outputs[0].bytes,
              "The consumer's retained packet differs from the full reference A output");
          }
          const auto matches_frame = [&](const stage_t &stage, std::size_t index) {
            return stage.copied && stage.bytes.size() == payload_size(r.frames[index]) &&
              std::memcmp(stage.bytes.data(), payload(r.frames[index]), stage.bytes.size()) == 0;
          };
          const auto &matched_b = records.borrowed_draw_sources[0];
          const auto &current_c = records.borrowed_draw_sources[1];
          require(matches_frame(matched_b, 1) && matches_frame(current_c, 0) && matches_frame(recycled_b_source, 0) &&
            matched_b.actual_resource_address == recycled_b_source.actual_resource_address &&
            matched_b.actual_resource_address != current_c.actual_resource_address &&
            (matched_b.source_misc & D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX) &&
            (current_c.source_misc & D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX),
            "Keyed snapshots did not retain literal B pixels across C arrival and B recycle");
          json artifacts = json::array();
          const auto save = [&](const std::string &name, const stage_t &stage) {
            const auto path = root / name;
            std::ofstream file(path, std::ios::binary);
            file.write(stage.bytes.data(), static_cast<std::streamsize>(stage.bytes.size()));
            file.flush();
            require(file.good(), "Borrowed capture readback artifact write failed");
            file.close();
            auto artifact = binding(path);
            artifact["role"] = name;
            artifact["resource_address"] = stage.actual_resource_address;
            return artifact;
          };
          const std::string suffix = r.hdr_input ? ".rgba16f" : ".bgra8";
          artifacts.push_back(save("late-draw-matched-B" + suffix, matched_b));
          artifacts.push_back(save("late-draw-current-C" + suffix, current_c));
          artifacts.push_back(save("recycled-B" + suffix, recycled_b_source));
          artifacts.push_back(save("late-draw-output" + (r.native_p010 ? ".p010" : native_nv12 ? ".nv12" : suffix), outputs[3]));
          if (r.native_p010) artifacts.push_back(save("matched-A.p010",outputs[0]));
          // A fresh C may have entered the sole slot during B's successful recovery. Retire
          // it on the same immutable C source before the diagnostic converter is destroyed.
          if (std::any_of(converter.matched_frame_slots.begin(), converter.matched_frame_slots.end(),
              [](const auto &slot) { return slot.pending; })) {
            wait_ready();
            convert(2);
            require(std::none_of(converter.matched_frame_slots.begin(), converter.matched_frame_slots.end(),
              [](const auto &slot) { return slot.pending; }), "Final C still retains a pending capture source");
          }
          write_json(root / "result.json", {{"schema",1},{"status",r.native_p010 ? "PASS_FUNCTIONAL_ACTUAL_CONVERTER_NATIVE_P010" : "PASS_FUNCTIONAL_ACTUAL_CONVERTER_BORROWED_CAPTURE_RECYCLE"},
            {"context_scope","synthetic-video-scheduling-only; not actual Win32 eligibility"}, {"source_inputs",sources},
            {"capture_device_adapter",capture_device_adapter},{"converter_device_adapter",converter_device_adapter},
            {"target_format",static_cast<UINT>(td.Format)},{"target_width",td.Width},{"target_height",td.Height},
            {"retained_rgb_target_untouched",check_retained_rgb},{"native_p010",native_p010_proof},
            {"initial_shutter_frame_direct_without_scratch",check_retained_rgb && !copy_reference},
            {"initial_shutter_output",check_retained_rgb ? save("initial-shutter-output" + suffix, initial_rgb_output) : json(nullptr)},
            {"late_draw_rgb_rendered_directly",recovered.rgb_rendered_directly},
            {"B_pending_pinned_source",true},{"B_key_timeout_preserved_ready_owner",true},{"B_released_and_recycled",true},
            {"C_and_recycled_B_use_archived_A_payload",true},{"private_matched_texture_allocated",false},
            {"before_cap",event_json(before)},{"key_timeout",event_json(key_timeout)},{"late_draw",event_json(*draw)},
            {"gate_held",gate_json(held_before)},{"gate_released",gate_json(released)},{"outputs",artifacts}});
          converter.depth_estimator.reset();est=nullptr;return 0;
        }
        std::uint64_t after_cap_conversion = 4u;
        if (native_nv12) {
          // Force the same input-cache invalidation used by stream gamma. The dirty input is
          // private to this fixture; neither packed RGB nor the held B publication is changed.
          const float overwrite_y[4] {1.0f, 0.0f, 0.0f, 0.0f};
          const float overwrite_uv[4] {};
          auto &output = converter.active_output_views();
          converter.device_ctx->ClearRenderTargetView(output.y_or_yuv.get(), overwrite_y);
          converter.device_ctx->ClearRenderTargetView(output.uv.get(), overwrite_uv);
          snapshot_target(4, "Sentinel NV12 input copy failed");
          converter.host_sbs_encoder_input_state.reset();
          convert(1);
          const auto redrawn = last_outcome(4u);
          require(redrawn.repeat && !redrawn.flat && !redrawn.queued && redrawn.route_matches &&
                  redrawn.pixel_source_frame == a_pixel_source_frame && redrawn.pixel_source_stamp_ticks == a_source_stamp &&
                  redrawn.current_source_stamp_ticks == before.current_source_stamp_ticks &&
                  converter.rendered_content_timestamp() == a_content_timestamp,
            "Input refresh did not redraw the same packed A while B remained pending");
          native_input_proof["cache_reset_repeat"] = event_json(redrawn);
          record_native_input("redrawn_A_after_cache_reset", redrawn, false);
          snapshot_target(5, "Redrawn NV12 input copy failed");
          (void) real_pending();
          ++after_cap_conversion;
        }
        std::this_thread::sleep_until(b_begin+std::chrono::milliseconds(280));
        convert(1); // The literal same B source; no new capture stamp or new root.
        const auto after=last_outcome(after_cap_conversion);
        const auto held_after=real_pending();
        require(after.flat&&!after.repeat&&!after.queued&&!after.packed_valid&&after.current_input&&
          after.views[0].Get()==after.current_input.Get()&&
          after.current_source_stamp_ticks==before.current_source_stamp_ticks&&
          after.slot_frames==before.slot_frames&&after.slot_observations==before.slot_observations&&
          after.slot_source_addresses==before.slot_source_addresses&&
          after.host_event_us-before.host_event_us>250000,
          "After-cap converter output was not actual current-flat B while B remained held");
        if (native_nv12) {
          record_native_input("current_flat_B_after_cap", after, false);
        }
        snapshot_target(2, "Actual after-cap flat target copy failed");
        // Independent GPU reference uses the production flat PS and recorder. It never writes
        // the converter target, chooses a converter route, or substitutes CPU-authored pixels.
        if (!native_nv12) {
          const auto b_context=converter.img_ctx_map.find(static_cast<img_d3d_t&>(*images[1]).id);
          require(b_context!=converter.img_ctx_map.end()&&b_context->second.encoder_mutex&&
            b_context->second.encoder_input_res.get()==after.current_input.Get(),"Actual B shared source identity unavailable");
          detail::keyed_mutex_lock_t source_lease {b_context->second.encoder_mutex.get()};
          require(source_lease.lock() == S_OK,"Independent B flat reference shared-source lease failed");
          host_sbs_v2_draw_command_t reference {
            .render_targets={flat_reference_rtv.Get(),nullptr},
            .vertex_shader=converter.sbs_reprojection_vs.get(),
            .pixel_shader=converter.sbs_flat_identity_ps.get(),
            .viewport=converter.sbs_viewport,
            .sampler=converter.sampler_linear.get(),
            .shader_resources={after.current_input.Get(),nullptr,nullptr,nullptr,nullptr,nullptr,nullptr},
            .geometry_constants=after.buffers[0].Get(),
          };
          require(record_host_sbs_v2_draw(converter.device_ctx.get(),reference),"Independent production flat reference draw failed");
          require(outputs[4].copy(converter.device_ctx.get(),flat_reference.Get()),"Independent flat reference target copy failed");
          converter.device_ctx->Flush(); // Submit shared reads before releasing their keyed lease.
        }
        // The independent timer releases 350 ms after real callback entry, even on render failure.
        wait_ready();
        const auto released=est->joined_completion_gate_status_for_benchmark();
        require(released&&released->released&&released->callback_exited&&!released->cancelled&&
          !released->callback_timed_out&&!released->callback_entry_timed_out,
          "B release did not follow the independent bounded gate timer");
        convert(1);
        const auto recovered=last_outcome(after_cap_conversion+1u);
        require(!recovered.flat&&!recovered.repeat&&!recovered.queued&&recovered.publication_current&&
          recovered.publication.frame_id==3u&&recovered.source_stamp_ticks==before.current_source_stamp_ticks&&
          recovered.current_source_stamp_ticks==before.current_source_stamp_ticks&&recovered.pending_slots==0u&&
          recovered.query_observed&&recovered.literal_last_query_result==CUDA_SUCCESS&&
          recovered.literal_ready_count>0u&&recovered.gate_released,
          "Exact held B did not genuinely ACK and recover without restamping/resubmission");
        require(recovered.views[0].Get() == b_context.encoder_input_res.get(),
          "Recovered B did not draw from its pinned shared source");
        check_released_b_pin();
        require(std::count_if(records.events.begin(),records.events.begin()+records.count,
          [](const auto &e){return e.kind==event_e::admission;})==2,
          "Static/pending/recovery deliveries unexpectedly admitted another root");
        if (native_nv12) {
          record_native_input("recovered_B", recovered, false);
        }
        snapshot_target(3, "Actual recovered B target copy failed");
        for (std::size_t i = 0; i < output_count; ++i) {
          outputs[i].read(converter.device_ctx.get());
          require(outputs[i].read_complete_us>=released->released_wall_us,"Target readback preceded real B release");
        }
        require(outputs[0].bytes == outputs[1].bytes, "Actual packed-before-cap output differs from retained A");
        constexpr std::array<const char *, 5u> rgb_names {{"matched-A.bgra8", "packed-A-before-cap.bgra8",
          "current-flat-B-after-cap.bgra8", "recovered-B.bgra8", "independent-flat-B-reference.bgra8"}};
        constexpr std::array<const char *, 6u> nv12_names {{"matched-A.nv12", "packed-A-before-cap.nv12",
          "current-flat-B-after-cap.nv12", "recovered-B.nv12", "sentinel-before-redraw.nv12", "packed-A-after-cache-reset.nv12"}};
        if (native_nv12) {
          const auto luma_bytes = static_cast<std::size_t>(td.Width) * td.Height;
          const auto &a_bytes = outputs[0].bytes;
          const auto &sentinel = outputs[4].bytes;
          require(a_bytes.size() == luma_bytes + luma_bytes / 2u && sentinel.size() == a_bytes.size(),
            "Native NV12 evidence has an invalid plane layout");
          require(!std::equal(a_bytes.begin(), a_bytes.begin() + luma_bytes, sentinel.begin()) &&
                  !std::equal(a_bytes.begin() + luma_bytes, a_bytes.end(), sentinel.begin() + luma_bytes),
            "Reset sentinel did not change both NV12 planes");
          require(outputs[5].bytes == a_bytes, "Reset repeat failed to restore both NV12 planes bit-for-bit");
          require(outputs[2].bytes != a_bytes && outputs[3].bytes != a_bytes,
            "Flat or recovered B is not distinguishable from retained NV12 A");
          native_input_proof["luma_bytes"] = luma_bytes;
          native_input_proof["chroma_bytes"] = luma_bytes / 2u;
          native_input_proof["repeat_preserved_both_planes"] = true;
          native_input_proof["cache_reset_redrew_both_planes"] = true;
        } else {
          require(outputs[2].bytes==outputs[4].bytes,"Actual after-cap RGB differs from independent production flat B GPU draw");
          require(outputs[0].bytes!=outputs[4].bytes,"A and flat B render identically; fixture cannot distinguish the source");
        }
        json artifacts=json::array(),trace=json::array();
        for (std::size_t i = 0; i < output_count; ++i) {
          const auto name = native_nv12 ? nv12_names[i] : rgb_names[i];
          const auto path=root/name;std::ofstream file(path,std::ios::binary);
          file.write(outputs[i].bytes.data(),static_cast<std::streamsize>(outputs[i].bytes.size()));file.flush();
          require(file.good(),"Continuity output artifact write failed");file.close();
          auto artifact=binding(path);artifact["role"]=name;
          artifact["copy_issued_us"]=outputs[i].copy_issued_us;artifact["read_complete_us"]=outputs[i].read_complete_us;
          artifacts.push_back(artifact);
        }
        for(std::size_t i=0;i<records.count;++i)trace.push_back(event_json(records.events[i]));
        auto qualified = json{{"schema",1},{"status", native_nv12 ? "PASS_FUNCTIONAL_ACTUAL_CONVERTER_NATIVE_NV12_INPUT_STABILITY" : "PASS_FUNCTIONAL_ACTUAL_CONVERTER_PACKED_CONTINUITY_ONLY"},
          {"context_scope","synthetic-video-scheduling-only; not actual Win32 eligibility"},
          {"source_inputs",sources},{"unused_third_source_not_admitted",true},
          {"static_idle_us",std::chrono::duration_cast<std::chrono::microseconds>(idle_end-idle_begin).count()},
          {"A_publication",pub(a_publication)},{"A_actual_RGB_pixel_source",a_pixel_source_frame},
          {"first_changed_diagnostics",result.at("first_changed_diagnostics")},
          {"before_cap",event_json(before)},{"after_cap",event_json(after)},
          {"recovered",event_json(recovered)},{"gate_before_cap",gate_json(held_before)},
          {"gate_after_cap",gate_json(held_after)},{"gate_released",gate_json(released)},
          {"gate_configuration",{{"placement",static_cast<UINT>(held_after.configuration.placement)},
            {"hold_ms",held_after.configuration.hold_after_callback_enter_ms},{"callback_safety_ms",held_after.configuration.callback_safety_timeout_ms}}},
          {"selected_adapter",selected_adapter},{"capture_device_adapter",capture_device_adapter},{"converter_device_adapter",converter_device_adapter},
          {"target_width",td.Width},{"target_height",td.Height},{"target_format",static_cast<UINT>(td.Format)},
          {"outputs",artifacts},{"trace",trace},{"readbacks_only_after_release",true},
          {"borrowed_capture",{{"A_draw_used_shared_source",true},{"B_pending_pinned_source",true},
            {"pool_skipped_B_while_pending",true},{"B_recovered_from_shared_source",true},
            {"pool_can_recycle_B_after_draw",true},{"private_matched_texture_allocated",false}}},
          {"no_CPU_warp_or_fabricated_completion_or_output_substitution",true},
          {"live_classification_encoder_presenter_latency_or_FPS_qualified",false}};
        if (native_nv12) {
          qualified["source_width"] = w;
          qualified["source_height"] = h;
          qualified["native_input_proof"] = std::move(native_input_proof);
        }
        write_json(root/"result.json", qualified);
        converter.depth_estimator.reset();est=nullptr;return 0;
      }
      // All diagnostic staging/textures/CPU vectors now exist. No Map/hash/JSON until release.
      records.armed=r.queued?1:-1;
      releaser=std::jthread([&](std::stop_token stop){const auto hard_deadline=clock_t::now()+std::chrono::milliseconds(500);while(!stop.stop_requested()){
        const bool proof_complete=release_after_draw_query.load(std::memory_order_acquire);
        const auto s=est->joined_completion_gate_status_for_benchmark();
        // The existing controller timer releases normally 120ms after callback entry.
        // A completed observation arms acceptance; it must NEVER cancel that normal release.
        if(clock_t::now()>=hard_deadline||(s&&(s->cancelled||s->callback_timed_out||s->callback_entry_timed_out))||(!proof_complete&&s&&s->released)){
          timeout_forced_release.store(true,std::memory_order_release);est->release_joined_completion_gate_for_benchmark();return;
        }
        if(proof_complete&&s&&s->callback_entered&&s->released&&s->callback_exited)return;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));}});
      stamp(1);const auto submit_begin=us();convert(1);const auto submit_return=us();converter.device_ctx->Flush();const auto flush_return=us();
      if(r.queued){require(records.captured[2],"No actual queued B draw on first miss");records.copy_target(2,target.Get());}
      const auto entered_deadline=clock_t::now()+std::chrono::seconds(2);
      for(;;){auto q=est->joined_completion_gate_status_for_benchmark();require(q&&!q->released&&clock_t::now()<entered_deadline,"No held B gate window");if(q->callback_entered)break;std::this_thread::sleep_for(std::chrono::milliseconds(1));}
      const auto pending=est->try_finish_pending_depth_nonblocking(r.hdr_input?models::input_color_space::scrgb_hdr:models::input_color_space::srgb,false);const auto after_query=us();
      const auto held=est->joined_completion_gate_status_for_benchmark();require(!pending.ready&&held&&held->callback_entered&&!held->released&&held->last_actual_event_query==CUDA_ERROR_NOT_READY,"B must be actual recorded NOT_READY");
      require(converter.sbs_frame_sequence==3&&std::count_if(converter.matched_frame_slots.begin(),converter.matched_frame_slots.end(),[](const auto &s){return s.pending&&s.frame_id==3;})==1,"Exact pending B source slot lost");
      const auto b_admission=std::find_if(records.events.begin(),records.events.begin()+records.count,[](const auto &e){return e.kind==event_e::admission&&e.conversion==3;});
      require(b_admission!=records.events.begin()+records.count&&b_admission->model_frame==3&&b_admission->force_infer&&!b_admission->gpu_undecided&&
        b_admission->optional_work==static_cast<std::uint32_t>(models::depth_optional_work_mode_e::ordinary),"B did not take the actual ordinary force-infer admission branch");
      if(r.queued)require(records.captured[1]&&records.captured[2]&&!records.phase_event[2].publication_current&&!records.phase_event[2].ticket_publication_current,"Guarded B draw without unpublished ticket proof");
      const bool a_current_after_queue=est->publication_is_current(a_publication);
      require(!r.queued||!a_current_after_queue,"Queued mutation retained old A geometry authority");
      std::optional<models::joined_completion_gate_benchmark::snapshot_t> pending_c_query;
      std::int64_t c_arrival_begin=0,c_arrival_return=0;
      if(pending_arrival){
        require(r.queued,"Pending-C trajectory requires the explicit queued actor");
        records.armed=4;stamp(2);c_arrival_begin=us();convert(2);c_arrival_return=us();records.copy_target(4,target.Get());converter.device_ctx->Flush();
        const auto still_pending=est->try_finish_pending_depth_nonblocking(r.hdr_input?models::input_color_space::scrgb_hdr:models::input_color_space::srgb,false);
        pending_c_query=est->joined_completion_gate_status_for_benchmark();
        require(!timeout_forced_release.load()&&!still_pending.ready&&pending_c_query&&pending_c_query->callback_entered&&!pending_c_query->released&&pending_c_query->last_actual_event_query==CUDA_ERROR_NOT_READY,"No real pending-B window at C return; timeout is unavailable, not success");
        require(records.captured[4]&&converter.sbs_frame_sequence==4&&std::count_if(converter.matched_frame_slots.begin(),converter.matched_frame_slots.end(),[](const auto&s){return s.pending&&s.frame_id==3;})==1,"C arrival lost actual pending B lease");
        require(std::none_of(records.events.begin(),records.events.begin()+records.count,[](const auto &e){return e.kind==event_e::admission&&e.conversion==4;}),"C was admitted while B retained the sole inflight lease");
      }
      require(!timeout_forced_release.load(),"Independent release deadline fired before selection proof");
      release_after_draw_query.store(true,std::memory_order_release);releaser.join();wait_ready();
      const auto released=est->joined_completion_gate_status_for_benchmark();require(released&&released->released,"Release not observed");released_before_readback=true;
      records.armed=3;const auto ack_begin=us();convert(pending_arrival?2:1);const auto ack_return=us();
      require(records.captured[3]&&records.phase_event[3].publication_current&&records.phase_event[3].publication.frame_id==3,"Genuine converter B ACK unavailable");
      if(!pending_arrival){require(converter.sbs_frame_sequence==4&&std::none_of(converter.matched_frame_slots.begin(),converter.matched_frame_slots.end(),[](const auto&s){return s.pending;}),"B ACK unexpectedly resubmitted");records.copy_target(3,target.Get());}
      if(!pending_arrival){records.armed=5;stamp(2);convert(2);}
      if(!records.captured[5]){wait_ready();convert(2);}
      require(records.captured[5]&&records.phase_event[5].publication_current,"Genuine C not yet drawn; no synthetic empty proof");
      records.copy_target(5,target.Get());records.read_all();
      for(std::size_t k=0;k<12;++k)require(records.stages[5][k].copied,"Genuine C core render tuple missing");
      if(r.queued){for(std::size_t k=0;k<slot_count;++k)if(records.stages[2][k].copied&&records.stages[3][k].copied){
        if(pending_arrival&&k==22){
          const auto &before=records.stages[2][k].bytes;const auto &after=records.stages[3][k].bytes;
          require(before.size()==r.frames[1].bgra.size()&&std::memcmp(before.data(),r.frames[1].bgra.data(),before.size())==0,"Queued B current input is not literal B");
          require(after.size()==r.frames[2].bgra.size()&&std::memcmp(after.data(),r.frames[2].bgra.data(),after.size())==0,"B ACK in current C call lost literal C input");
        } else require(records.stages[2][k].bytes==records.stages[3][k].bytes,"Same-instance queued/ordinary B byte mismatch");
        if(k<12)require(records.stages[2][k].actual_resource_address==records.stages[3][k].actual_resource_address,"Queued mutable working alias not authenticated by real B ACK");}}
      if(pending_arrival&&records.phase_event[4].repeat){
        const auto &e=records.phase_event[4];require(e.actual_draw_pixel_source==0&&e.pixel_source_frame==3&&e.watermark_before==e.watermark_after,"Packed B was mislabeled as C draw/watermark");
        require(records.stages[4][20].bytes==records.stages[2][20].bytes&&records.stages[4][21].bytes==records.stages[2][21].bytes,"Pending C did not retain literal queued B pixels");
        require(records.stages[4][22].bytes.size()==r.frames[2].bgra.size()&&std::memcmp(records.stages[4][22].bytes.data(),r.frames[2].bgra.data(),r.frames[2].bgra.size())==0,"Pending-C current input is not literal C");
      }
      require(std::all_of(records.stages[5][3].bytes.begin(),records.stages[5][3].bytes.end(),[](char v){return v==0;}),"Actual current C M not zero");
      const auto word=[&](std::size_t slot,std::size_t offset){const auto &v=records.stages[5][slot].bytes;require(v.size()>=(offset+1)*4,"C typed packet truncated");std::uint32_t n=0;std::memcpy(&n,v.data()+offset*4,4);return n;};
      const auto &ce=records.phase_event[5];
      require(word(6,0)==5&&word(6,1)==0x35544353u&&word(6,22)==27&&word(6,23)==1,"C SCT is not a current valid producer packet");
      require((std::uint64_t(word(6,4))|(std::uint64_t(word(6,5))<<32))==ce.model_frame&&
        (std::uint64_t(word(6,20))|(std::uint64_t(word(6,21))<<32))==ce.observation_us,"C SCT clock is not own current model observation");
      require(word(7,0)==4&&word(7,1)==0x3852434fu&&word(7,2)==1&&word(7,3)==0&&word(7,4)==0,"C own OCR does not report current zero detections");
      require(word(7,5)==word(6,4)&&word(7,6)==word(6,5)&&word(8,20)==word(6,3),"C OCR/UIP/SCT identity join failed");
      release();converter.probe_observer={};converter.probe_context=nullptr;
      json captures=json::array();for(std::size_t p=0;p<phase_count;++p)if(records.captured[p]){
        auto dir=root/phases[p];std::filesystem::create_directories(dir);json artifacts=json::object();
        for(std::size_t k=0;k<slot_count;++k){const auto &s=records.stages[p][k];if(!s.copied)continue;const std::string name=r.hdr_input&&(k==0u||k==20u||k==21u||k==22u)?std::string(names[k]).replace(std::string(names[k]).size()-6u,6u,".rgba16f"):names[k];const auto path=dir/name;std::ofstream f(path,std::ios::binary);f.write(s.bytes.data(),static_cast<std::streamsize>(s.bytes.size()));f.flush();require(f.good(),"Raw artifact write failed");f.close();auto d=binding(path);d["row_bytes"]=s.row_bytes;d["rows"]=s.rows;d["format"]=static_cast<unsigned>(s.td.Format);d["kind"]=static_cast<unsigned>(s.kind);d["actual_resource_address"]=s.actual_resource_address;d["actual_structure_stride"]=s.source_stride;d["actual_bind_flags"]=s.source_bind;d["actual_misc_flags"]=s.source_misc;d["copy_issued_us"]=s.copy_issued_us;d["read_complete_us"]=s.read_complete_us;artifacts[name]=d;}
        auto j=event_json(records.phase_event[p]);j["artifacts"]=artifacts;j["phase"]=phases[p];write_json(dir/"phase.json",j);captures.push_back(binding(dir/"phase.json"));}
      json trace=json::array();for(std::size_t i=0;i<records.count;++i)trace.push_back(event_json(records.events[i]));
      result={{"schema",1},{"status","COMPLETE"},{"queue_enabled",r.queued},{"actual_converter",true},{"NVENC_or_capture_session",false},
        {"research_context_is_not_real_Win32_video_eligibility",true},{"semantic_UI_EMPTY_authority",false},{"source_inputs",sources},{"selected_matte_enabled",r.selected_matte.has_value()},{"selected_current_outline",records.selected_outline},{"input_transfer",r.hdr_input?2u:0u},{"capture_and_target_format",r.hdr_input?10u:87u},{"fixture_HDR_declaration",r.hdr_input},{"background_observer_required",r.background_observer},
        {"adapter_selection","EXPLICIT_LUID_NO_FALLBACK"},{"required_adapter_luid",{required_luid_low,required_luid_high}},
        {"enumerated_adapters",adapter_entries},{"selected_adapter",selected_adapter},{"capture_device_adapter",capture_device_adapter},{"converter_device_adapter",converter_device_adapter},
        {"source_stamp_kind","fixture-source-admission steady clock; not DDup/presenter clock"},{"model_clock_formula","floor(actual copied_at steady microseconds)+1"},
        {"steady_period_num",clock_t::period::num},{"steady_period_den",clock_t::period::den},{"traces",trace},{"phases",captures},
        {"B_submit_begin_us",submit_begin},{"B_submit_return_us",submit_return},{"B_Flush_return_us",flush_return},{"B_after_actual_query_us",after_query},
        {"B_ACK_begin_us",ack_begin},{"B_ACK_return_us",ack_return},{"gate_held",gate_json(held)},{"gate_released",gate_json(released)},
        {"pending_C_trajectory",pending_arrival},{"C_arrival_begin_us",c_arrival_begin},{"C_arrival_return_us",c_arrival_return},{"gate_after_C_arrival",gate_json(pending_c_query)},{"timeout_forced_release",timeout_forced_release.load()},
        {"old_A_publication",pub(a_publication)},{"old_A_current_after_queue",a_current_after_queue},{"readback_only_after_release",released_before_readback},
        {"same_instance_queued_vs_B_ACK_tuple",r.queued},{"ordinary_B_draw_reference_exercised",!pending_arrival},{"late_B_after_C_draw_exercised",pending_arrival&&records.phase_event[4].actual_draw_pixel_source!=0},{"current_C_may_legally_repeat_B_until_actual_C_draw",true},
        {"live_classification_encoder_presenter_latency_or_FPS_qualified",false},{"no_CPU_warp_or_fabricated_completion",true}};
      write_json(root/"result.json",result);
      // All current work is genuinely ACKed and copied. Finish this one canonical model lease
      // before a second SEQUENTIAL converter instance; no parallel model/stream architecture.
      converter.depth_estimator.reset();est=nullptr;return 0;
    } catch(const std::exception &e) {
      error=e.what();release();converter.probe_observer={};converter.probe_context=nullptr;
      if(created)try{json trace=json::array();for(std::size_t i=0;i<records.count;++i)trace.push_back(event_json(records.events[i]));write_json(root/"failure.json",{{"schema",1},{"status","UNAVAILABLE_OR_FAILED_NO_QUALIFICATION"},{"error",error},{"gate",est?gate_json(est->joined_completion_gate_status_for_benchmark()):json(nullptr)},{"partial_observer_events",trace},{"phase_diagnostics",result},{"timeout_forced_release",timeout_forced_release.load()},{"enumerated_adapters",adapter_entries},{"selected_adapter",selected_adapter},{"capture_device_adapter",capture_device_adapter},{"converter_device_adapter",converter_device_adapter},{"no_generated_success",true}});}catch(...){}
      return 7;
    }
  }
  int run(const request_t &r,std::string &error) noexcept {
    if (r.native_p010) {
      // One TensorRT/graphics/native owner at a time, with the original A/B gate protocol.
      for (int codec : {1,2}) {
        const auto code = run_one(r,false,error,codec);if(code)return code;
      }
      return 0;
    }
    const int first=run_one(r,false,error);if(first)return first;
    if(r.queued)return run_one(r,true,error);
    return 0;
  }
}

// PRIVATE fixed-six ordinary converter diagnostic; the original A/B/C scheduler is untouched.
namespace actual_converter_probe_v1 {
  namespace {
    struct sequence6_recorder_t {
      std::vector<event_t> events;
      bool overflow=false;
      static void record(void *p,const event_t &e) noexcept {
        auto &s=*static_cast<sequence6_recorder_t*>(p);
        if(s.events.size()==s.events.capacity()){s.overflow=true;return;}
        s.events.push_back(e); // reserved before native admission; storage only, no verdict
      }
      const event_t *own_draw(std::size_t start,std::int64_t stamp) const {
        for(std::size_t i=events.size();i>start;--i) {
          const auto &e=events[i-1u];
          if(e.kind==event_e::draw && e.publication_current && !e.queued && !e.repeat && !e.flat &&
             e.source_stamp_ticks==stamp && e.current_source_stamp_ticks==stamp &&
             e.model_frame==e.publication.frame_id && e.observation_us==e.publication.observation_timestamp_us &&
             e.actual_draw_pixel_source==e.conversion) return &e;
        }
        return nullptr;
      }
    };
    json sequence6_copy_matched(ID3D11Device *device,ID3D11DeviceContext *ctx,
      const event_t &e,ID3D11Texture2D *target,const std::filesystem::path &dir,
      std::size_t ordinal,const frame_t &input) {
      std::array<stage_t,slot_count> stages;
      const auto copy=[&](std::size_t k,ID3D11Resource *r) {
        if(!r)return;stages[k].prepare(device,r);require(stages[k].copy(ctx,r),"Fixed-six passive copy failed");
      };
      for(std::size_t k=0;k<12u;++k){auto r=resource(e.views[k].Get());copy(k,r.Get());}
      for(std::size_t k=0;k<8u;++k)copy(12u+k,e.buffers[k].Get());
      copy(20u,e.packed.Get());copy(21u,target);auto current=resource(e.current_input.Get());copy(22u,current.Get());
      constexpr std::array<std::size_t,11u> selected_map{23u,24u,25u,26u,27u,28u,29u,30u,32u,33u,34u};
      for(std::size_t k=0;k<selected_map.size();++k){auto r=resource(e.selected_views[k].Get());copy(selected_map[k],r.Get());}
      copy(31u,e.selected_contour_constants.Get());
      for(std::size_t k=0;k<e.current_outline_stages.size();++k){auto r=resource(e.current_outline_stages[k].Get());copy(39u+k,r.Get());}
      // Every copy above is ordered before the next image can be admitted. Maps/hash are
      // functional diagnostics only; no inference/draw predicate reads these staging views.
      ctx->Flush();for(auto &s:stages)s.read(ctx);
      // Preserve literal native bytes before any route/join gate can report failure.
      std::filesystem::create_directories(dir);json artifacts=json::object(),absent=json::array();
      for(std::size_t k=0;k<slot_count;++k) {
        const auto &s=stages[k];if(!s.copied){absent.push_back(names[k]);continue;}
        const auto path=dir/names[k];require(!std::filesystem::exists(path),"Fixed-six refuses artifact overwrite");
        std::ofstream f(path,std::ios::binary);f.write(s.bytes.data(),static_cast<std::streamsize>(s.bytes.size()));
        f.flush();require(f.good(),"Fixed-six raw output write failed");f.close();auto d=binding(path);
        d["row_bytes"]=s.row_bytes;d["rows"]=s.rows;d["format"]=static_cast<unsigned>(s.td.Format);
        d["kind"]=static_cast<unsigned>(s.kind);d["actual_resource_address"]=s.actual_resource_address;
        d["actual_structure_stride"]=s.source_stride;d["actual_bind_flags"]=s.source_bind;d["actual_misc_flags"]=s.source_misc;
        d["copy_issued_us"]=s.copy_issued_us;d["read_complete_us"]=s.read_complete_us;artifacts[names[k]]=d;
      }
      auto out=event_json(e);out["ordinal_SOURCE_ORDER_ONLY"]=ordinal;out["declared_PNG_sha256"]=input.source_file_sha256;
      out["declared_timeline_us_METADATA_ONLY"]=input.declared_timeline_us;out["artifacts"]=artifacts;out["absent_slots_NOT_ZERO"]=absent;
      out["M_slot3_role"]="original estimator legacy M; actual successful selected draw M is slot25";
      out["Source5_raw_proof64_and_envelope104_exported"]=false;
      out["Source5_evidence_scope"]="actual joined raw parent, display snapshot, current C24/P/S/SCT/UIP/root plus analysis band and selected GPU receipt; no invented proof array";
      write_json(dir/"phase.json",out);
      require(e.selected_draw && e.selected_views[0] && e.selected_views[1] && e.selected_views[2],
        "Own matched draw did not bind selected current output; fallback is not a kind3 proof");
      constexpr std::array<std::size_t,24u> required{0u,3u,4u,6u,7u,8u,15u,16u,22u,23u,24u,25u,29u,30u,31u,39u,40u,41u,42u,43u,44u,45u,46u,47u};
      for(auto k:required)require(stages[k].copied,"Fixed-six mandatory actual operand missing");
      require(stages[23u].bytes==stages[25u].bytes,"Actual draw M differs from selected M");
      require(stages[23u].actual_resource_address==stages[25u].actual_resource_address,
        "Actual PS t3 resource differs from the selected output resource");
      require(stages[24u].actual_resource_address==e.actual_draw_selected_receipt_address,
        "Actual draw selected receipt resource differs from copied diagnostic receipt");
      require(stages[22u].bytes.size()==input.bgra.size() &&
        std::memcmp(stages[22u].bytes.data(),input.bgra.data(),input.bgra.size())==0,
        "Own matched current source is not the literal decoded PNG");
      const auto word=[&](std::size_t k,std::size_t j){
        require(stages[k].bytes.size()>=(j+1u)*4u,"Typed actual output truncated");
        std::uint32_t v=0;std::memcpy(&v,stages[k].bytes.data()+j*4u,4u);return v;
      };
      require(stages[24u].bytes.size()==256u && stages[31u].bytes.size()==96u &&
        stages[6u].bytes.size()==128u && stages[7u].bytes.size()==1024u && stages[8u].bytes.size()==160u,
        "Actual selected/C24/SCT/OCR/UIP ABI mismatch");
      require(word(24u,2u)==3u && word(24u,3u)==0u && word(24u,62u)==2u,
        "Actual selected receipt is not current kind3/fault0/algorithm2");
      for(std::size_t j=0;j<24u;++j)require(word(24u,4u+j)==word(31u,j),"Selected receipt/C24 join failed");
      for(std::size_t j=0;j<32u;++j)require(word(24u,28u+j)==word(6u,j),"Selected receipt/SCT join failed");
      require((std::uint64_t(word(24u,60u))|(std::uint64_t(word(24u,61u))<<32u))==e.publication.transaction_token,
        "Selected receipt/publication token mismatch");
      return binding(dir/"phase.json");
    }
  }
  int run_sequence6(const sequence6_request_t &q,std::string &error) noexcept {
    const auto &r=q.common;error.clear();std::shared_ptr<fixture_display_t> display;d3d_base_encode_device converter;
    sequence6_recorder_t records;ComPtr<ID3D11Texture2D> target;ComPtr<ID3D11RenderTargetView> target_rtv;
    std::array<std::shared_ptr<platf::img_t>,6u> images;models::video_depth_estimator *est=nullptr;
    json sources=json::array(),captures=json::array(),adapter_entries=json::array(),selected_adapter=nullptr,
      capture_device_adapter=nullptr,converter_device_adapter=nullptr;bool created=false;
    const auto root=r.output/"actual-converter-sequence6-sdr-v1";
    try {
      require(r.enabled && r.explicit_adapter && !r.queued && !r.hdr_input && !r.background_observer &&
        !r.temporal_source_bridge && r.selected_matte && r.selected_matte->current_outline,
        "Fixed-six requires explicit ordinary SDR current-outline request only");
      records.events.reserve(1024u); // bounded native event storage, before any source admission
      require(!std::filesystem::exists(root),"Fixed-six refuses output overwrite");
      const UINT w=q.frames[0].width,h=q.frames[0].height;
      require(w&&h&&w<=5120u&&h<=5120u,"Fixed-six invalid source extent");
      for(const auto &f:q.frames)require(f.width==w&&f.height==h&&f.bgra.size()==std::size_t(w)*h*4u&&
        f.rgba16.empty()&&f.source_file_sha256.size()==64u,"Fixed-six source shape/binding mismatch");
      std::filesystem::create_directories(root);created=true;
      for(std::size_t i=0;i<6u;++i){const auto path=root/("input_"+std::to_string(i+1u)+".bgra8");
        std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(q.frames[i].bgra.data()),static_cast<std::streamsize>(q.frames[i].bgra.size()));
        f.flush();require(f.good(),"Fixed-six input write failed");f.close();sources.push_back({{"raw",binding(path)},
          {"declared_PNG_sha256",q.frames[i].source_file_sha256},{"declared_timeline_us_METADATA_ONLY",q.frames[i].declared_timeline_us}});}
      require(models::prepare_tensorrt_model(SUNSHINE_ASSETS_DIR,::video::host_sbs_v2_depth_model(),config::video.adapter_name),"Resident depth preparation failed");
      require(models::prepare_ocr_tensorrt_model(SUNSHINE_ASSETS_DIR,config::video.adapter_name),"Resident OCR preparation failed");
      if(!sbs_reprojection_vs_hlsl)require(::platf::dxgi::init()==0,"Platform shader initialization failed");
      ComPtr<IDXGIFactory1> factory;hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)),"Factory creation failed");
      const DWORD required_luid_low=r.hdr_adapter_luid_low;const LONG required_luid_high=r.hdr_adapter_luid_high;
      require((required_luid_low!=0u||required_luid_high!=0)&&r.hdr_adapter_vendor_id&&r.hdr_adapter_device_id,"Explicit fixed-six adapter identity missing");
      const auto adapter_json=[](const DXGI_ADAPTER_DESC1 &d){return json{{"vendor_id",d.VendorId},{"device_id",d.DeviceId},
        {"luid_low",d.AdapterLuid.LowPart},{"luid_high",d.AdapterLuid.HighPart},{"flags",d.Flags}};};
      ComPtr<IDXGIAdapter1> adapter;UINT matches=0;DXGI_ADAPTER_DESC1 adapter_desc{};
      for(UINT i=0;;++i){ComPtr<IDXGIAdapter1>a;const auto status=factory->EnumAdapters1(i,&a);if(status==DXGI_ERROR_NOT_FOUND)break;hr(status,"Adapter enumeration failed");
        DXGI_ADAPTER_DESC1 d{};hr(a->GetDesc1(&d),"Adapter desc failed");auto row=adapter_json(d);row["index"]=i;adapter_entries.push_back(row);
        if(d.AdapterLuid.LowPart==required_luid_low&&d.AdapterLuid.HighPart==required_luid_high){adapter=a;adapter_desc=d;selected_adapter=row;++matches;}}
      require(matches==1u&&adapter_desc.VendorId==r.hdr_adapter_vendor_id&&adapter_desc.DeviceId==r.hdr_adapter_device_id&&
        !(adapter_desc.Flags&DXGI_ADAPTER_FLAG_SOFTWARE),"Explicit fixed-six LUID unavailable/ambiguous; no fallback");
      const auto actual_device_adapter=[&](ID3D11Device *device){ComPtr<IDXGIDevice> dxgi;hr(device->QueryInterface(IID_PPV_ARGS(&dxgi)),"Actual DXGI device failed");
        ComPtr<IDXGIAdapter> base;hr(dxgi->GetAdapter(&base),"Actual adapter failed");ComPtr<IDXGIAdapter1>a;hr(base.As(&a),"Actual adapter1 failed");
        DXGI_ADAPTER_DESC1 d{};hr(a->GetDesc1(&d),"Actual adapter desc failed");require(d.AdapterLuid.LowPart==required_luid_low&&
          d.AdapterLuid.HighPart==required_luid_high&&d.VendorId==adapter_desc.VendorId&&d.DeviceId==adapter_desc.DeviceId,"Actual device differs from required adapter");return adapter_json(d);};
      display=std::make_shared<fixture_display_t>();display->width=display->width_before_rotation=w;display->height=display->height_before_rotation=h;
      display->env_width=w;display->env_height=h;display->display_rotation=DXGI_MODE_ROTATION_IDENTITY;
      display->capture_format=DXGI_FORMAT_B8G8R8A8_UNORM;display->declared_hdr=false;display->next_image_id=1;
      const D3D_FEATURE_LEVEL levels[]{D3D_FEATURE_LEVEL_11_1,D3D_FEATURE_LEVEL_11_0};
      hr(D3D11CreateDevice(adapter.Get(),D3D_DRIVER_TYPE_UNKNOWN,nullptr,D3D11_CREATE_DEVICE_BGRA_SUPPORT,levels,2,D3D11_SDK_VERSION,
        &display->device,nullptr,&display->device_ctx),"Fixed-six capture device failed");capture_device_adapter=actual_device_adapter(display->device.get());
      require(converter.init(display,adapter.Get(),pix_fmt_e::nv12)==0,"Real converter init failed");converter_device_adapter=actual_device_adapter(converter.device.get());
      const context_t context{w,h,0x3651455300000001ull};converter.probe_context=&context;
      auto settings=r.settings;settings.reshade=false;require(converter.init_rgb_output(w*2u,h,::video::SBS_AI,settings)==0,"Real RGB init failed");
      const auto prepare_deadline=clock_t::now()+std::chrono::seconds(60);
      while(!converter.ensure_depth_estimator()){require(clock_t::now()<prepare_deadline&&converter.host_sbs_renderer!=models::host_sbs_renderer_e::failed_flat,"Estimator preparation unavailable");std::this_thread::sleep_for(std::chrono::milliseconds(1));}
      est=converter.depth_estimator.get();require(est&&est->is_valid(),"Real estimator unavailable");std::string setup_error;
      require(converter.configure_selected_matte_for_instance(*r.selected_matte,setup_error),("Fixed-six current-outline preparation: "+setup_error).c_str());
      D3D11_TEXTURE2D_DESC td{};td.Width=2u*w;td.Height=h;td.MipLevels=1;td.ArraySize=1;td.Format=DXGI_FORMAT_B8G8R8A8_UNORM;
      td.SampleDesc.Count=1;td.Usage=D3D11_USAGE_DEFAULT;td.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;
      hr(converter.device->CreateTexture2D(&td,nullptr,&target),"Fixed-six target failed");hr(converter.device->CreateRenderTargetView(target.Get(),nullptr,&target_rtv),"Fixed-six target RTV failed");
      for(std::size_t i=0;i<6u;++i){images[i]=display->alloc_img();require(images[i]&&display->complete_img(images[i].get(),false)==0,"Shared image allocation failed");
        auto &img=static_cast<img_d3d_t&>(*images[i]);detail::keyed_mutex_lock_t lock {img.capture_mutex.get()};require(lock.lock() == S_OK,"Capture keyed mutex failed");
        display->device_ctx->UpdateSubresource(img.capture_texture.get(),0u,nullptr,q.frames[i].bgra.data(),w*4u,0u);display->device_ctx->Flush();img.blank=false;}
      for(const auto &image:images){auto &img=static_cast<img_d3d_t&>(*image);auto [it,inserted]=converter.img_ctx_map.try_emplace(img.id);(void)inserted;
        require(converter.initialize_image_context(img,it->second)==0,"Shared image preparation failed");}
      D3D11_TEXTURE2D_DESC private_desc{};static_cast<img_d3d_t&>(*images[0]).capture_texture->GetDesc(&private_desc);
      private_desc.Usage=D3D11_USAGE_DEFAULT;private_desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;private_desc.MiscFlags=0;private_desc.CPUAccessFlags=0;
      for(auto &slot:converter.matched_frame_slots){hr(converter.device->CreateTexture2D(&private_desc,nullptr,&slot.texture),"Private slot preparation failed");hr(converter.device->CreateShaderResourceView(slot.texture.get(),nullptr,&slot.srv),"Private slot SRV failed");}
      converter.probe_observer={&records,&sequence6_recorder_t::record};
      for(std::size_t i=0;i<6u;++i) {
        const auto stamp=clock_t::now();images[i]->frame_timestamp=stamp;images[i]->content_timestamp=stamp;
        const auto ticks=stamp.time_since_epoch().count();const auto first_event=records.events.size();
        const auto deadline=clock_t::now()+std::chrono::seconds(30);const event_t *own=nullptr;std::size_t conversions=0;
        while(!own && conversions<8u) {
          require(clock_t::now()<deadline&&!est->has_terminal_failure(),"Fixed-six own ACK timeout or terminal estimator failure");
          require(converter.convert_rgb(*images[i],target.Get(),target_rtv.Get(),false,clock_t::now()-std::chrono::milliseconds(1))==d3d_base_encode_device::rgb_conversion_e::converted,"Actual fixed-six convert failed");++conversions;
          require(!records.overflow,"Fixed-six bounded event recorder overflow");own=records.own_draw(first_event,ticks);if(own)break;
          while(!est->can_accept_frame()){require(clock_t::now()<deadline&&!est->has_terminal_failure(),"Fixed-six native readiness timeout");std::this_thread::sleep_for(std::chrono::milliseconds(1));}
        }
        require(own,"Fixed-six own matched draw unavailable after bounded retries");
        require(std::none_of(converter.matched_frame_slots.begin(),converter.matched_frame_slots.end(),
          [](const auto &slot){return slot.pending;}),"Own matched draw left another pending source; no next admission");
        const auto actual=*own;auto &img=static_cast<img_d3d_t&>(*images[i]);
        const auto image_context=converter.img_ctx_map.find(img.id);
        require(image_context!=converter.img_ctx_map.end(),"Own matched encoder image context missing");
        detail::keyed_mutex_lock_t lease {image_context->second.encoder_mutex.get()};
        require(lease.lock() == S_OK,"Own matched encoder-device diagnostic keyed lease failed");
        const auto dir=root/("observation_"+std::to_string(i+1u));
        captures.push_back(sequence6_copy_matched(converter.device.get(),converter.device_ctx.get(),actual,target.Get(),dir,i+1u,q.frames[i]));
        converter.device_ctx->Flush(); // all shared reads submitted before releasing this actual image lease
      }
      converter.probe_observer={};converter.probe_context=nullptr;json trace=json::array();for(const auto &e:records.events)trace.push_back(event_json(e));
      write_json(root/"result.json",{{"schema",1},{"status","COMPLETE_FIXED_SIX_ACTUAL_CURRENT_DRAWS"},{"actual_converter",true},
        {"source_inputs",sources},{"phases",captures},{"traces",trace},{"selected_current_outline",true},{"capture_and_target_format",87u},
        {"input_transfer",0u},{"required_adapter_luid",{r.hdr_adapter_luid_low,r.hdr_adapter_luid_high}},
        {"enumerated_adapters",adapter_entries},{"selected_adapter",selected_adapter},{"capture_device_adapter",capture_device_adapter},{"converter_device_adapter",converter_device_adapter},
        {"context_scope","synthetic-video-scheduling-only; not actual Win32 eligibility"},{"source_stamp_kind","one actual steady-clock stamp per immutable source; unchanged on ACK retries"},
        {"model_clock_formula","floor(actual copied_at steady microseconds)+1; CLI timeline metadata only"},
        {"scheduler","one owner, each source polled to its own real matched draw before next admission; no held gate/queue experiment"},
        {"event_capacity",1024u},{"max_conversions_per_source",8u},{"per_source_timeout_seconds",30u},
        {"streaming_cadence_or_latency_claim",false},{"GPU_cost_claim",false},{"new_component_snapshot_hooks",false},{"no_authored_P_S_C24_or_mask_inputs",true}});
      converter.depth_estimator.reset();est=nullptr;return 0;
    }catch(const std::exception &e){error=e.what();converter.probe_observer={};converter.probe_context=nullptr;
      if(created)try{json trace=json::array();for(const auto &v:records.events)trace.push_back(event_json(v));
        write_json(root/"failure.json",{{"schema",1},{"status","UNAVAILABLE_OR_FAILED_NO_QUALIFICATION"},{"error",error},
          {"completed_phases",captures},{"partial_native_events",trace},{"no_generated_success",true}});}catch(...){}
      converter.depth_estimator.reset();est=nullptr;return 7;}
  }
}
