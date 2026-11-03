// Included only inside platf::dxgi AFTER the actual converter class is complete.
// No duplicate converter policy, CUDA verdict, CPU warp or encoder session.
int init();
namespace pending_current_colour_probe_v1 {
  using namespace actual_converter_probe_v1;
  namespace {
    using Microsoft::WRL::ComPtr;
    using json = nlohmann::json;
    using clock_t = std::chrono::steady_clock;
    constexpr std::size_t phase_count=7, slot_count=61;
    constexpr std::array<const char *,phase_count> phases{{"matched_A","before_queue_B_old_working","queued_B_before_ACK","B_after_genuine_ACK","C_arrival_while_B_pending","matched_C","same_C_capture_while_B_pending"}};
    constexpr std::array<const char *,slot_count> names{{"draw-source.bgra8","Base.f32","state.u32","M.u32","display-source.rgba16f","SCP.u32","SCT.u32","OCR.u32","UIP.u32","cut.u32","frame-state.u32","EMA.u32","b2.u32","b3.u32","b5.u32","transaction.u32","arguments.u32","depth-CB.u32","region-CB.u32","contour-CB.u32","packed.bgra8","rgb-target.bgra8","current-input.bgra8","C-display-source.rgba16f","C-source-receipt.u32","C-source-context.u32","negative-expected.u32","negative-receipt.u32","owner-bank0-source.rgba16f","owner-bank0-P.f32","owner-bank0-OCR.u32","owner-bank0-UIP.u32","owner-bank0-SCT.u32","owner-bank0-context.u32","owner-bank0-transaction.u32","owner-bank0-arguments.u32","owner-bank0-owner.u32","owner-bank0-anchors.u32","owner-bank0-scratch.u32","owner-bank0-scores.u32","owner-bank0-receipt.u32","owner-bank0-expected.u32","owner-bank0-watermark.u32","owner-bank0-root-expected.u32","owner-bank1-source.rgba16f","owner-bank1-P.f32","owner-bank1-OCR.u32","owner-bank1-UIP.u32","owner-bank1-SCT.u32","owner-bank1-context.u32","owner-bank1-transaction.u32","owner-bank1-arguments.u32","owner-bank1-owner.u32","owner-bank1-anchors.u32","owner-bank1-scratch.u32","owner-bank1-scores.u32","owner-bank1-receipt.u32","owner-bank1-expected.u32","owner-bank1-watermark.u32","owner-bank1-root-expected.u32","heldBase-currentC-reference.bgra8"}};
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
        {"queued",e.queued},{"flat",e.flat},{"repeat",e.repeat},{"cached_current_colour",e.cached_current_colour},{"pending_current_colour",e.pending_current_colour},{"material_epoch",e.material_epoch},{"material_serial",e.material_serial},{"material_capture_clock_us",e.material_capture_clock_us},{"material_interval_contiguous",e.material_interval_contiguous},{"negative_bank",e.negative_bank},
        {"slot_frames",e.slot_frames},{"slot_pending",e.slot_pending},{"slot_observations",e.slot_observations},{"slot_contexts",e.slot_contexts},
        {"slot_source_addresses",e.slot_source_addresses},{"selected_slot",e.selected_slot},{"pending_slots",e.pending_slots},
        {"publication_current",e.publication_current},{"ticket_publication_current",e.ticket_publication_current},{"publication",pub(e.publication)},
        {"planned_publication_VALUE_ONLY",pub(e.planned_publication)},{"actual_draw_pixel_source",e.actual_draw_pixel_source},
        {"pixel_source_frame",e.pixel_source_frame},{"pixel_source_stamp_ticks",e.pixel_source_stamp_ticks},{"packed_valid",e.packed_valid},
        {"packed_age_us",e.packed_age_us},{"route_matches",e.route_matches},{"source_matches",e.source_matches},
        {"watermark_before",e.watermark_before},{"watermark_after",e.watermark_after},
        {"literal_query_observed",e.query_observed},{"literal_last_query_result",e.literal_last_query_result},{"literal_query_count",e.literal_query_count},
        {"literal_query_target_frame",e.literal_query_target_frame},
        {"literal_not_ready_count",e.literal_not_ready_count},{"literal_ready_count",e.literal_ready_count},{"gate_entered",e.gate_entered},{"gate_released",e.gate_released}};
    }
    struct fixture_display_t final : display_vram_t {
      bool is_hdr() override { return false; } // declared SDR fixture, no fake IDXGIOutput
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
          const UINT pixel=td.Format==DXGI_FORMAT_R16G16B16A16_FLOAT?8u:4u;
          require(pixel==8u||td.Format==DXGI_FORMAT_R32_FLOAT||td.Format==DXGI_FORMAT_R32_UINT||td.Format==DXGI_FORMAT_B8G8R8A8_UNORM,"Unsupported staging format");
          auto desc=td;desc.Usage=D3D11_USAGE_STAGING;desc.BindFlags=0;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;desc.MiscFlags=0;
          ComPtr<ID3D11Texture2D>out;hr(d->CreateTexture2D(&desc,nullptr,&out),"Staging texture allocation failed");staging=out;
          row_bytes=std::size_t(td.Width)*pixel;rows=td.Height;bytes.resize(row_bytes*rows);}
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
      int armed=-1;bool prepared=false,pending_arrival=false;
      std::array<ComPtr<ID3D11Buffer>,3> queued_B_constant_aliases;
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
        const std::array<ComPtr<ID3D11Resource>,5> extra{{resource(copy.negative_current_source.Get()),resource(copy.negative_current_receipt.Get()),copy.negative_current_constants,copy.negative_expected,resource(copy.negative_receipt.Get())}};
        for(std::size_t i=0;i<5u;++i)if(extra[i]&&!stages[phase][23u+i].copy(ctx,extra[i].Get()))capture_fault=true;
        for(std::size_t bank=0;bank<2u;++bank)for(std::size_t i=0;i<16u;++i)if(copy.negative_owner_resources[bank][i]&&!stages[phase][28u+16u*bank+i].copy(ctx,copy.negative_owner_resources[bank][i].Get()))capture_fault=true;
        phase_event[phase]=copy;captured[phase]=true;
      }
      static void record(void *state,const event_t &e) noexcept {
        auto &s=*static_cast<recorder_t*>(state);if(s.count==s.events.size()){s.overflow=true;return;}s.events[s.count++]=e;
        if(e.kind==event_e::draw&&e.publication_current&&!e.queued)for(std::size_t i=7;i<12;++i)s.old_working[i-7]=e.views[i];
        if(s.armed==1&&e.kind==event_e::queue_begin)s.capture(1,e);
        if(s.armed==1&&e.kind==event_e::draw&&e.queued)s.capture(2,e);
        if(s.armed==3&&e.publication_current&&!e.queued&&e.publication.frame_id==3&&
           ((s.pending_arrival&&e.kind==event_e::ack)||(!s.pending_arrival&&e.kind==event_e::draw))){s.capture(3,e);if(s.pending_arrival)s.armed=5;}
        if(s.armed==6&&e.conversion==5&&(e.kind==event_e::packed_repeat||e.kind==event_e::draw))s.capture(6,e);
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
        }prepared=true;capture(0,a);require(!capture_fault,"A capture setup failed");
      }
      void prepare_negative(ID3D11Device *device,const models::research_pending_current_colour_v1::resources_t &lane,ID3D11Texture2D *reference) {
        const auto current=lane.current_snapshot();
        for(auto &phase:stages) {
          phase[23].prepare(device,resource(current.source.Get()).Get());phase[24].prepare(device,current.receipt_buffer.Get());phase[25].prepare(device,lane.current_constants().Get());
          phase[26].prepare_buffer(device,160u);phase[27].prepare_buffer(device,256u);
          for(UINT bank=0;bank<2u;++bank){const auto operands=lane.owner_resources(bank);for(UINT i=0;i<16u;++i)phase[28u+16u*bank+i].prepare(device,operands[i].Get());}
          phase[60].prepare(device,reference);
        }
      }
      void copy_target(std::size_t phase,ID3D11Texture2D *target){require(stages[phase][21].copy(ctx,target),"RGB target copy failed");}
      void read_all(){for(auto &p:stages)for(auto &s:p)s.read(ctx);}
    };
  }

  int run_one(const request_t &r,bool pending_arrival,std::string &error) noexcept {
    error.clear();std::shared_ptr<fixture_display_t> display;d3d_base_encode_device converter;
    recorder_t records;ComPtr<ID3D11Texture2D> target,reference;ComPtr<ID3D11RenderTargetView> target_rtv,reference_rtv;
    ComPtr<ID3D11Buffer> reference_subtitle_disabled;
    std::array<std::shared_ptr<platf::img_t>,3> images;
    models::video_depth_estimator *est=nullptr;
    std::jthread releaser;std::atomic<bool> release_after_draw_query{false},timeout_forced_release{false};
    records.pending_arrival=pending_arrival;
    json result,adapter_entries=json::array(),selected_adapter=nullptr,capture_device_adapter=nullptr,converter_device_adapter=nullptr;bool created=false,released_before_readback=false;
    const auto root=r.output/(pending_arrival?"trajectory2-pending-C":"trajectory1-B-parity")/"actual-converter-pending-current-colour-v1";
    const auto release=[&](){if(est)est->release_joined_completion_gate_for_benchmark();release_after_draw_query.store(true);if(releaser.joinable()){releaser.request_stop();releaser.join();}};
    try {
      require(r.enabled&&r.queued&&r.pending_current_colour.enabled,"Explicit pending-colour probe not enabled");require(!std::filesystem::exists(root),"Probe refuses overwrite");
      const UINT w=r.frames[0].width,h=r.frames[0].height;
      require(w&&h&&w<=5120u&&h<=5120u,"Invalid source extent");
      for(const auto &f:r.frames)require(f.width==w&&f.height==h&&f.bgra.size()==std::size_t(w)*h*4&&f.source_file_sha256.size()==64,"Source shape/binding mismatch");
      std::filesystem::create_directories(root);created=true;
      json sources=json::array();for(std::size_t i=0;i<3;++i){auto p=root/("input_"+std::to_string(i)+".bgra8");std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(r.frames[i].bgra.data()),static_cast<std::streamsize>(r.frames[i].bgra.size()));f.flush();require(f.good(),"Input raw write failed");f.close();sources.push_back({{"raw",binding(p)},{"declared_PNG_sha256",r.frames[i].source_file_sha256},{"declared_timeline_us_METADATA_ONLY",r.frames[i].declared_timeline_us}});}
      require(models::prepare_tensorrt_model(SUNSHINE_ASSETS_DIR,::video::host_sbs_v2_depth_model(),config::video.adapter_name),"Resident depth preparation failed");
      require(models::prepare_ocr_tensorrt_model(SUNSHINE_ASSETS_DIR,config::video.adapter_name),"Resident OCR preparation failed");
      if(!sbs_reprojection_vs_hlsl)require(::platf::dxgi::init()==0,"Platform shader initialization failed");
      ComPtr<IDXGIFactory1> factory;hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)),"Factory creation failed");
      // Explicit fixture-machine identity, bound to the saved prior hardware diagnostic and
      // read-only DXGI census. Never choose the first NVIDIA entry or a name-only match.
      constexpr DWORD required_luid_low=3780540602u;
      constexpr LONG required_luid_high=0;
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
      require(matches==1&&adapter_desc.VendorId==0x10de&&!(adapter_desc.Flags&DXGI_ADAPTER_FLAG_SOFTWARE),"Explicit fixture NVIDIA LUID unavailable or ambiguous; no fallback");
      const auto actual_device_adapter=[&](ID3D11Device *device){
        ComPtr<IDXGIDevice> dxgi;hr(device->QueryInterface(IID_PPV_ARGS(&dxgi)),"Actual device DXGI query failed");
        ComPtr<IDXGIAdapter> base;hr(dxgi->GetAdapter(&base),"Actual device GetAdapter failed");ComPtr<IDXGIAdapter1> a;hr(base.As(&a),"Actual adapter1 query failed");
        DXGI_ADAPTER_DESC1 d{};hr(a->GetDesc1(&d),"Actual device adapter desc failed");
        require(d.AdapterLuid.LowPart==required_luid_low&&d.AdapterLuid.HighPart==required_luid_high&&d.VendorId==adapter_desc.VendorId&&d.DeviceId==adapter_desc.DeviceId,
          "Actual device adapter differs from explicit selected LUID");return adapter_json(d);
      };
      display=std::make_shared<fixture_display_t>();display->width=display->width_before_rotation=w;display->height=display->height_before_rotation=h;
      display->env_width=w;display->env_height=h;display->display_rotation=DXGI_MODE_ROTATION_IDENTITY;display->capture_format=DXGI_FORMAT_B8G8R8A8_UNORM;display->next_image_id=1;
      const D3D_FEATURE_LEVEL levels[]{D3D_FEATURE_LEVEL_11_1,D3D_FEATURE_LEVEL_11_0};
      hr(D3D11CreateDevice(adapter.Get(),D3D_DRIVER_TYPE_UNKNOWN,nullptr,D3D11_CREATE_DEVICE_BGRA_SUPPORT,levels,2,D3D11_SDK_VERSION,&display->device,nullptr,&display->device_ctx),"Capture-side fixture device failed");
      capture_device_adapter=actual_device_adapter(display->device.get());
      require(converter.init(display,adapter.Get(),pix_fmt_e::nv12)==0,"Real converter init failed");
      converter_device_adapter=actual_device_adapter(converter.device.get());
      const context_t context{w,h,0x4356524700000001ull};converter.probe_context=&context;
      auto settings=r.settings;settings.reshade=false; // only authored source, not an external provider
      require(converter.init_rgb_output(w*2,h,::video::SBS_AI,settings)==0,"Real RGB-only initialization failed");
      const auto prepare_deadline=clock_t::now()+std::chrono::seconds(60);
      while(!converter.ensure_depth_estimator()){require(clock_t::now()<prepare_deadline&&converter.host_sbs_renderer!=models::host_sbs_renderer_e::failed_flat,"Estimator preparation unavailable");std::this_thread::sleep_for(std::chrono::milliseconds(1));}
      est=converter.depth_estimator.get();require(est&&est->is_valid(),"Real estimator unavailable");
      require(est->set_completed_ocr_conditioner_research_enabled(true),"Preinput warm descriptors unavailable");
      if(r.queued)require(converter.initialize_queued_renderer_for_instance(true),"Per-instance queued resources unavailable");
      models::joined_completion_gate_benchmark::configuration_t gate;gate.enabled=true;gate.placement=models::joined_completion_gate_benchmark::placement_e::before_interop_unmaps;
      gate.target_frame_id=3;gate.hold_after_callback_enter_ms=120;gate.callback_safety_timeout_ms=500;
      require(est->configure_joined_completion_gate_for_benchmark(gate),"Pre-input gate configuration failed");
      D3D11_TEXTURE2D_DESC td{};td.Width=2*w;td.Height=h;td.MipLevels=1;td.ArraySize=1;td.Format=DXGI_FORMAT_B8G8R8A8_UNORM;td.SampleDesc.Count=1;td.Usage=D3D11_USAGE_DEFAULT;td.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;
      hr(converter.device->CreateTexture2D(&td,nullptr,&target),"RGB target failed");hr(converter.device->CreateRenderTargetView(target.Get(),nullptr,&target_rtv),"RGB target RTV failed");
      hr(converter.device->CreateTexture2D(&td,nullptr,&reference),"Reference target failed");hr(converter.device->CreateRenderTargetView(reference.Get(),nullptr,&reference_rtv),"Reference RTV failed");
      models::host_sbs_subtitle_render_constants_t disabled_subtitle{};D3D11_BUFFER_DESC disabled_desc{};disabled_desc.ByteWidth=sizeof(disabled_subtitle);disabled_desc.Usage=D3D11_USAGE_DEFAULT;disabled_desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;D3D11_SUBRESOURCE_DATA disabled_init{&disabled_subtitle,0,0};
      hr(converter.device->CreateBuffer(&disabled_desc,&disabled_init,&reference_subtitle_disabled),"Reference UI CB failed");
      for(std::size_t i=0;i<3;++i){images[i]=display->alloc_img();require(images[i]&&display->complete_img(images[i].get(),false)==0,"Actual shared image allocation failed");auto &img=static_cast<img_d3d_t&>(*images[i]);texture_lock_helper lock(img.capture_mutex.get());require(lock.lock(),"Capture keyed mutex failed");display->device_ctx->UpdateSubresource(img.capture_texture.get(),0,nullptr,r.frames[i].bgra.data(),w*4,0);display->device_ctx->Flush();img.blank=false;}
      // Open every existing shared-image binding and allocate the existing TWO private slots
      // before input1. Allocation establishes no frame, source clock, pending or publication.
      for(const auto &image:images){auto &img=static_cast<img_d3d_t&>(*image);auto [it,inserted]=converter.img_ctx_map.try_emplace(img.id);(void)inserted;require(converter.initialize_image_context(img,it->second)==0,"Shared image preparation failed");}
      D3D11_TEXTURE2D_DESC private_desc{};static_cast<img_d3d_t&>(*images[0]).capture_texture->GetDesc(&private_desc);
      private_desc.Usage=D3D11_USAGE_DEFAULT;private_desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;private_desc.MiscFlags=0;private_desc.CPUAccessFlags=0;
      for(auto &slot:converter.matched_frame_slots){hr(converter.device->CreateTexture2D(&private_desc,nullptr,&slot.texture),"Private slot preparation failed");hr(converter.device->CreateShaderResourceView(slot.texture.get(),nullptr,&slot.srv),"Private slot SRV failed");}
      records.ctx=converter.device_ctx.get();converter.probe_observer={&records,&recorder_t::record};
      const auto stamp=[&](std::size_t i){const auto t=clock_t::now();images[i]->frame_timestamp=t;images[i]->content_timestamp=t;};
      const auto convert=[&](std::size_t i){require(converter.convert_rgb(*images[i],target.Get(),target_rtv.Get(),false,clock_t::now()-std::chrono::milliseconds(1))==0,"Actual RGB convert failed");require(!records.overflow&&!records.capture_fault,"Passive observer unavailable");};
      const auto wait_ready=[&](){const auto until=clock_t::now()+std::chrono::seconds(30);while(!est->can_accept_frame()){require(!est->has_terminal_failure()&&clock_t::now()<until,"Real root readiness timeout");std::this_thread::sleep_for(std::chrono::milliseconds(1));}};
      stamp(0);convert(0);wait_ready();convert(0); // exact same img/source/content identity on retry
      const auto *a=records.last_draw();require(a&&a->publication_current&&!a->queued&&a->publication.frame_id==1&&converter.sbs_frame_sequence==2,"A/ACK protocol unavailable; no invented ordinal");
      require(std::none_of(converter.matched_frame_slots.begin(),converter.matched_frame_slots.end(),[](const auto &s){return s.pending;}),"A retry unexpectedly enqueued; target3 unavailable");
      {
        // convert() has released its encoder-side keyed lease. Reacquire that SAME
        // prepared A context before the initial diagnostic reads its shared SRV.
        const auto a_context=converter.img_ctx_map.find(static_cast<img_d3d_t&>(*images[0]).id);
        require(a_context!=converter.img_ctx_map.end()&&a_context->second.encoder_mutex&&
          a_context->second.encoder_input_res.get()==a->current_input.Get(),"Prepared A shared context identity unavailable");
        texture_lock_helper a_snapshot_lease(a_context->second.encoder_mutex.get());
        require(a_snapshot_lease.lock(),"Initial A shared diagnostic keyed lease failed");
        records.prepare(converter.device.get(),*a,target.Get());records.copy_target(0,target.Get());
        converter.device_ctx->Flush(); // submit every shared read BEFORE ReleaseSync(0)
      }
      const auto a_publication=a->publication;
      require(converter.initialize_pending_current_colour_for_instance(r.pending_current_colour,a_publication,error),"Pending-colour resources unavailable before B");
      records.prepare_negative(converter.device.get(),converter.pending_colour_lane,reference.Get());
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
      const auto pending=est->try_finish_pending_depth_nonblocking(models::input_color_space::srgb,false);const auto after_query=us();
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
        const auto still_pending=est->try_finish_pending_depth_nonblocking(models::input_color_space::srgb,false);
        pending_c_query=est->joined_completion_gate_status_for_benchmark();
        require(!timeout_forced_release.load()&&!still_pending.ready&&pending_c_query&&pending_c_query->callback_entered&&!pending_c_query->released&&pending_c_query->last_actual_event_query==CUDA_ERROR_NOT_READY,"No real pending-B window at C return; timeout is unavailable, not success");
        require(records.captured[4]&&converter.sbs_frame_sequence==4&&std::count_if(converter.matched_frame_slots.begin(),converter.matched_frame_slots.end(),[](const auto&s){return s.pending&&s.frame_id==3;})==1,"C arrival lost actual pending B lease");
        require(std::none_of(records.events.begin(),records.events.begin()+records.count,[](const auto &e){return e.kind==event_e::admission&&e.conversion==4;}),"C was admitted while B retained the sole inflight lease");
        require(records.phase_event[4].pending_current_colour&&!records.phase_event[4].repeat&&records.phase_event[4].actual_draw_pixel_source==4u,"Actual current-C negative-only draw unavailable");
        const auto first_c_identity=records.phase_event[4];
        records.armed=6;convert(2);records.copy_target(6,target.Get());converter.device_ctx->Flush();
        require(records.captured[6]&&records.phase_event[6].material_epoch==first_c_identity.material_epoch&&records.phase_event[6].material_serial==first_c_identity.material_serial&&records.phase_event[6].material_capture_clock_us==first_c_identity.material_capture_clock_us,"Same capture retry minted a material observation");
        const auto c_context=converter.img_ctx_map.find(static_cast<img_d3d_t&>(*images[2]).id);
        require(c_context!=converter.img_ctx_map.end()&&c_context->second.encoder_mutex,"Current C encoder binding missing");
        {texture_lock_helper lease(c_context->second.encoder_mutex.get());require(lease.lock(),"Reference current-C keyed lease failed");
          const auto &e=records.phase_event[4];host_sbs_v2_draw_command_t command;
          command.render_targets[0]=reference_rtv.Get();command.vertex_shader=converter.sbs_reprojection_vs.get();command.pixel_shader=converter.sbs_reprojection_v2_live_ps.get();
          command.viewport=converter.sbs_viewport;command.sampler=converter.sampler_linear.get();
          command.shader_resources={c_context->second.encoder_input_res.get(),e.views[1].Get(),e.views[2].Get(),nullptr,nullptr,nullptr,nullptr};command.geometry_constants=e.buffers[0].Get();command.subtitle_constants=reference_subtitle_disabled.Get();
          require(record_host_sbs_v2_draw(converter.device_ctx.get(),command),"Independent GPU heldBase/current-C reference draw failed");
          require(records.stages[4][60].copy(converter.device_ctx.get(),reference.Get()),"Independent reference copy failed");converter.device_ctx->Flush();}
        const auto after_reference=est->try_finish_pending_depth_nonblocking(models::input_color_space::srgb,false);const auto still=est->joined_completion_gate_status_for_benchmark();
        require(!after_reference.ready&&still&&still->last_actual_event_query==CUDA_ERROR_NOT_READY&&!still->released,"No actual B600 after current C/reference draw recording");
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
      if(r.queued){for(std::size_t k=0;k<23u;++k)if(records.stages[2][k].copied&&records.stages[3][k].copied){
        if(pending_arrival&&k==20){
          require(records.stages[3][20].bytes==records.stages[4][20].bytes,"Late B ACK rolled packed current C back");
        } else if(pending_arrival&&k==22){
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
      if(pending_arrival){const auto &e=records.phase_event[4];
        require(e.material_serial>0&&e.material_capture_clock_us>0&&!e.publication_current&&!e.ticket_publication_current,"C draw promoted model/publication authority");
        require(e.watermark_after>=4u&&records.phase_event[3].watermark_after>=4u,"Late ACK lost current C watermark");
        require(records.stages[4][20].bytes==records.stages[6][20].bytes,"Same C capture changed output pixels");
        const auto &negative=records.stages[4][27].bytes;require(negative.size()==256u,"Negative receipt missing");
        std::uint32_t status=0,fault=0;std::memcpy(&status,negative.data()+8,4);std::memcpy(&fault,negative.data()+12,4);
        require(fault==0u&&status==2u,"Actual B-to-C negative pattern unavailable; UNKNOWN is not kickout success");
        require(records.stages[4][20].bytes==records.stages[4][60].bytes,"Retired overlay differs from independent GPU heldBase/current-C reference");
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
        for(std::size_t k=0;k<slot_count;++k){const auto &s=records.stages[p][k];if(!s.copied)continue;const auto path=dir/names[k];std::ofstream f(path,std::ios::binary);f.write(s.bytes.data(),static_cast<std::streamsize>(s.bytes.size()));f.flush();require(f.good(),"Raw artifact write failed");f.close();auto d=binding(path);d["row_bytes"]=s.row_bytes;d["rows"]=s.rows;d["format"]=static_cast<unsigned>(s.td.Format);d["kind"]=static_cast<unsigned>(s.kind);d["actual_resource_address"]=s.actual_resource_address;d["actual_structure_stride"]=s.source_stride;d["actual_bind_flags"]=s.source_bind;d["actual_misc_flags"]=s.source_misc;d["copy_issued_us"]=s.copy_issued_us;d["read_complete_us"]=s.read_complete_us;artifacts[names[k]]=d;}
        auto j=event_json(records.phase_event[p]);j["artifacts"]=artifacts;j["phase"]=phases[p];write_json(dir/"phase.json",j);captures.push_back(binding(dir/"phase.json"));}
      json trace=json::array();for(std::size_t i=0;i<records.count;++i)trace.push_back(event_json(records.events[i]));
      result={{"schema",1},{"status","COMPLETE"},{"queue_enabled",r.queued},{"actual_converter",true},{"NVENC_or_capture_session",false},
        {"research_context_is_not_real_Win32_video_eligibility",true},{"semantic_UI_EMPTY_authority",false},{"source_inputs",sources},
        {"adapter_selection","EXPLICIT_LUID_NO_FALLBACK"},{"required_adapter_luid",{required_luid_low,required_luid_high}},
        {"enumerated_adapters",adapter_entries},{"selected_adapter",selected_adapter},{"capture_device_adapter",capture_device_adapter},{"converter_device_adapter",converter_device_adapter},
        {"source_stamp_kind","fixture-source-admission steady clock; not DDup/presenter clock"},{"model_clock_formula","floor(actual copied_at steady microseconds)+1"},
        {"steady_period_num",clock_t::period::num},{"steady_period_den",clock_t::period::den},{"traces",trace},{"phases",captures},
        {"B_submit_begin_us",submit_begin},{"B_submit_return_us",submit_return},{"B_Flush_return_us",flush_return},{"B_after_actual_query_us",after_query},
        {"B_ACK_begin_us",ack_begin},{"B_ACK_return_us",ack_return},{"gate_held",gate_json(held)},{"gate_released",gate_json(released)},
        {"pending_C_trajectory",pending_arrival},{"C_arrival_begin_us",c_arrival_begin},{"C_arrival_return_us",c_arrival_return},{"gate_after_C_arrival",gate_json(pending_c_query)},{"timeout_forced_release",timeout_forced_release.load()},
        {"old_A_publication",pub(a_publication)},{"old_A_current_after_queue",a_current_after_queue},{"readback_only_after_release",released_before_readback},
        {"same_instance_queued_vs_B_ACK_tuple",r.queued},{"ordinary_B_draw_reference_exercised",!pending_arrival},{"late_B_after_C_draw_exercised",pending_arrival&&records.phase_event[4].actual_draw_pixel_source!=0},{"current_C_may_legally_repeat_B_until_actual_C_draw",false},{"negative_only_pending_current_colour",true},{"currentC_glyph_EMPTY_or_depth_authority",false},
        {"live_classification_encoder_presenter_latency_or_FPS_qualified",false},{"no_CPU_warp_or_fabricated_completion",true}};
      write_json(root/"result.json",result);
      // All current work is genuinely ACKed and copied. Finish this one canonical model lease
      // before a second SEQUENTIAL converter instance; no parallel model/stream architecture.
      converter.depth_estimator.reset();est=nullptr;return 0;
    } catch(const std::exception &e) {
      error=e.what();release();converter.probe_observer={};converter.probe_context=nullptr;
      if(created)try{json trace=json::array();for(std::size_t i=0;i<records.count;++i)trace.push_back(event_json(records.events[i]));write_json(root/"failure.json",{{"schema",1},{"status","UNAVAILABLE_OR_FAILED_NO_QUALIFICATION"},{"error",error},{"gate",est?gate_json(est->joined_completion_gate_status_for_benchmark()):json(nullptr)},{"partial_observer_events",trace},{"timeout_forced_release",timeout_forced_release.load()},{"enumerated_adapters",adapter_entries},{"selected_adapter",selected_adapter},{"capture_device_adapter",capture_device_adapter},{"converter_device_adapter",converter_device_adapter},{"no_generated_success",true}});}catch(...){}
      return 7;
    }
  }
  int run(const request_t &r,std::string &error) noexcept {
    const int first=pending_current_colour_probe_v1::run_one(r,false,error);if(first)return first;
    if(r.queued)return pending_current_colour_probe_v1::run_one(r,true,error);
    return 0;
  }
}
