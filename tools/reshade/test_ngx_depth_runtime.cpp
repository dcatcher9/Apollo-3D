// SPDX-License-Identifier: GPL-3.0-only
// Actual exported NGX calls -> production adapter -> native D3D12 copy/fences ->
// shared adaptive controller -> the official ReShade HDR shader. The synthetic
// SDK reports metadata only; no capture, depth binding or readiness is injected.
#define SUNSHINE_DIRECT_RUNTIME_FIXTURE_ONLY
#include "test_streamline_direct_runtime.cpp"
#include <atomic>
#include <d3d12sdklayers.h>
#include <limits>

namespace ngx_fixture {
  constexpr std::uint32_t success=1, failure=0xbad00001u;
  struct parameters {
    ID3D12Resource *depth{};
    unsigned width{},height{},left{},top{},active_width{},active_height{};
    int flags=8,reset=0;
    bool evaluation_success{},provide_depth=true,provide_extent=true;
  };
  std::array<unsigned,16> handle_storage{};
  unsigned created{},evaluated{},released{},getters{},next_handle{};
  const void *expected_handle{};
}

#if defined(_MSC_VER)
#define NGX_FIXTURE_EXPORT extern "C" __declspec(dllexport) __declspec(noinline)
#else
#define NGX_FIXTURE_EXPORT extern "C" __declspec(dllexport) __attribute__((noinline,noipa))
#endif

// Export the real C names and signatures. Runtime module discovery and MinHook
// must find these ordinary game-side SDK entry points without any test adapter.
NGX_FIXTURE_EXPORT std::uint32_t __cdecl NVSDK_NGX_D3D12_CreateFeature(
    void *commands,std::uint32_t feature,void *parameters,void **handle) {
  if(!commands || feature!=1 || !parameters || !handle || ngx_fixture::next_handle>=ngx_fixture::handle_storage.size())
    return ngx_fixture::failure;
  *handle=&ngx_fixture::handle_storage[ngx_fixture::next_handle++];
  ngx_fixture::expected_handle=*handle;
  ++ngx_fixture::created;
  return ngx_fixture::success;
}
NGX_FIXTURE_EXPORT std::uint32_t __cdecl NVSDK_NGX_D3D12_EvaluateFeature(
    void *commands,const void *handle,const void *parameters,void (__cdecl *)(float,bool &)) {
  ++ngx_fixture::evaluated;
  if(!commands || handle!=ngx_fixture::expected_handle || !parameters) return ngx_fixture::failure;
  return static_cast<const ngx_fixture::parameters *>(parameters)->evaluation_success ? ngx_fixture::success : ngx_fixture::failure;
}
NGX_FIXTURE_EXPORT std::uint32_t __cdecl NVSDK_NGX_D3D12_ReleaseFeature(void *handle) {
  if(handle!=ngx_fixture::expected_handle) return ngx_fixture::failure;
  ngx_fixture::expected_handle=nullptr;
  ++ngx_fixture::released;
  return ngx_fixture::success;
}
NGX_FIXTURE_EXPORT std::uint32_t __cdecl NVSDK_NGX_Parameter_GetI(
    const void *parameters,const char *name,int *out) {
  ++ngx_fixture::getters;
  if(!parameters || !name || !out) return ngx_fixture::failure;
  const auto &value=*static_cast<const ngx_fixture::parameters *>(parameters);
  if(!std::strcmp(name,"DLSS.Feature.Create.Flags")) *out=value.flags;
  else if(!std::strcmp(name,"Reset")) *out=value.reset;
  else return ngx_fixture::failure;
  return ngx_fixture::success;
}
NGX_FIXTURE_EXPORT std::uint32_t __cdecl NVSDK_NGX_Parameter_GetUI(
    const void *parameters,const char *name,unsigned *out) {
  ++ngx_fixture::getters;
  if(!parameters || !name || !out) return ngx_fixture::failure;
  const auto &value=*static_cast<const ngx_fixture::parameters *>(parameters);
  if(!std::strcmp(name,"Width")) *out=value.width;
  else if(!std::strcmp(name,"Height")) *out=value.height;
  else if(!std::strcmp(name,"DLSS.Render.Subrect.Dimensions.Width") && value.provide_extent) *out=value.active_width;
  else if(!std::strcmp(name,"DLSS.Render.Subrect.Dimensions.Height") && value.provide_extent) *out=value.active_height;
  else if(!std::strcmp(name,"DLSS.Input.Depth.Subrect.Base.X")) *out=value.left;
  else if(!std::strcmp(name,"DLSS.Input.Depth.Subrect.Base.Y")) *out=value.top;
  else return ngx_fixture::failure;
  return ngx_fixture::success;
}
NGX_FIXTURE_EXPORT std::uint32_t __cdecl NVSDK_NGX_Parameter_GetD3d12Resource(
    const void *parameters,const char *name,ID3D12Resource **out) {
  ++ngx_fixture::getters;
  if(!parameters || !name || !out || std::strcmp(name,"Depth")) return ngx_fixture::failure;
  const auto &value=*static_cast<const ngx_fixture::parameters *>(parameters);
  if(!value.provide_depth) return ngx_fixture::failure;
  *out=value.depth;
  return ngx_fixture::success;
}

namespace {
  std::function<void()> ngx_frame_observer;
  void *cached_fg_options{}; // Captured before ReShade/add-on discovery in the live-compatibility case.
  void observe_ngx_source(api::effect_runtime *runtime,api::effect_technique technique,
      api::command_list *commands,api::resource_view rtv,api::resource_view rtv_srgb) {
    observe_source(runtime,technique,commands,rtv,rtv_srgb);
    char name[256]{};runtime->get_technique_name(technique,name);
    if(named(name,technique_name) && ngx_frame_observer) ngx_frame_observer();
  }
  struct ngx_runtime_fixture : direct_fixture {
    com_ptr<ID3D12InfoQueue> debug_messages;
    ngx_fixture::parameters parameters;
    void *feature{};
    unsigned active_width{},active_height{},left{},top{};
    bool armed{},reversed=true,invalid_extent{},use_depth_stencil_crop{},auto_create=true;
    std::unique_ptr<uav_target> depth_stencil_crop;
    std::vector<std::unique_ptr<uav_target>> pressure_resources;
    bool state_pressure{};
    unsigned pressure_recordings{};
    bool cross_queue{},complete_producer{},producer_gated{},retire_producer_recording=true,submit_producer_recording=true;
    bool independent_generated_present{};
    bool producer_oracle_only{},skip_ngx_gated{};
    bool tracked_source_test{},tracked_source{},parity_scene{},evaluate_without_source_state{};
    std::unique_ptr<uav_target> tracked_decoy;
    com_ptr<ID3D12RootSignature> tracked_root;
    com_ptr<ID3D12PipelineState> tracked_pipeline;
    com_ptr<ID3D12CommandQueue> producer_queue;
    com_ptr<ID3D12CommandAllocator> producer_allocator;
    com_ptr<ID3D12CommandAllocator> producer_retirement_allocator;
    com_ptr<ID3D12GraphicsCommandList> producer_commands;
    com_ptr<ID3D12Fence> producer_completion;
    std::uint64_t producer_fence_value{};
    std::uint64_t producer_recording_native{};
    unsigned producer_submissions{},producer_generation{};
    std::function<void()> record_fg_tags;
    std::uint64_t logical_source{};
    decltype(&NVSDK_NGX_D3D12_CreateFeature) volatile call_create{};
    decltype(&NVSDK_NGX_D3D12_EvaluateFeature) volatile call_evaluate{};
    decltype(&NVSDK_NGX_D3D12_ReleaseFeature) volatile call_release{};
    const uav_target &ngx_source() const { return use_depth_stencil_crop ? *depth_stencil_crop : *selected; }
    bool packed_source() const { return tracked_source || use_depth_stencil_crop; }
    float tracked_clear_value() const { return parity_scene ? (reversed ? 0.f : 1.f) : (reversed ? .9375f : .0625f); }

    std::uint64_t capture_demand(bool expected) const {
      if(!sunshine_camera_fixture::flag("SUNSHINE_CAPTURE_DEMAND_TEST")) return 0;
      using demand_t=BOOL (*)(api::effect_runtime *,BOOL *,unsigned *,std::uint64_t *);
      const auto query=reinterpret_cast<demand_t>(GetProcAddress(GetModuleHandleW(L"SunshineSBSTest.addon64"),"SunshineDepthTestCaptureDemand"));
      BOOL enabled{};unsigned users{};std::uint64_t transitions{};
      require(query && query(observed.runtime,&enabled,&users,&transitions),"Capture-demand regression requires its passive observation hook");
      require(bool(enabled)==expected && users==unsigned(expected),
        "Generic capture demand disagrees with the active native/shared/manual capture owner");
      return transitions;
    }

    void report_debug_messages() {
      if(!debug_messages.p)return;
      const auto total=debug_messages->GetNumStoredMessagesAllowedByRetrievalFilter();
      const auto first=total>256?total-256:0;
      std::printf("MEASURE D3D12 debug messages stored=%llu inspect_last=%llu discarded=%llu (at most32 warnings/errors)\n",
        static_cast<unsigned long long>(total),static_cast<unsigned long long>(total-first),
        static_cast<unsigned long long>(debug_messages->GetNumMessagesDiscardedByMessageCountLimit()));
      unsigned reported{};
      for(auto index=first;index<total && reported<32;++index) {
        SIZE_T bytes{};
        if(FAILED(debug_messages->GetMessage(index,nullptr,&bytes)) || bytes<sizeof(D3D12_MESSAGE) || bytes>1024*1024)continue;
        std::vector<std::uint8_t> storage(bytes);
        auto *message=reinterpret_cast<D3D12_MESSAGE *>(storage.data());
        if(FAILED(debug_messages->GetMessage(index,message,&bytes)) || message->Severity>D3D12_MESSAGE_SEVERITY_WARNING)continue;
        std::printf("D3D12 DEBUG severity=%u id=%u category=%u: %.*s\n",unsigned(message->Severity),unsigned(message->ID),
          unsigned(message->Category),int(std::min<SIZE_T>(message->DescriptionByteLength,1024)),message->pDescription);
        ++reported;
      }
    }

    void run_frame_generation() {
      namespace wire = reshade_bridge;
      using namespace sunshine_streamline;
      struct fg_options { base_structure base; std::uint32_t mode{}, generated_frames{}; };
      static_assert(sizeof(fg_options)==40);
      using set_options_t=std::int32_t (*)(const abi_v2::viewport &,const fg_options &);
      const auto interposer=GetModuleHandleW(L"sl.interposer.dll");
      const auto get_function=reinterpret_cast<abi_v2::get_feature_function>(GetProcAddress(interposer,"slGetFeatureFunction"));
      require(interposer && get_function,"Frame-generation fixture SDK was not loaded");
      const bool live_compat=sunshine_camera_fixture::flag("SUNSHINE_FG_LIVE_COMPAT_TEST");
      const bool late_source_test=sunshine_camera_fixture::flag("SUNSHINE_FG_LATE_SOURCE_TEST");
      producer_oracle_only=sunshine_camera_fixture::flag("SUNSHINE_FG_PRODUCER_ORACLE_ONLY_TEST");
      skip_ngx_gated=sunshine_camera_fixture::flag("SUNSHINE_FG_SKIP_NGX_GATED_TEST");
      const bool contiguous_copy=sunshine_camera_fixture::flag("SUNSHINE_FG_CONTIGUOUS_COPY_TEST");
      const bool tag_fallback_test=sunshine_camera_fixture::flag("SUNSHINE_SL_TAG_FALLBACK_TEST");
      enum class high_tag_case { absent, unsupported, usable, neither_usable };
      auto high_tag_mode=tag_fallback_test?high_tag_case::unsupported:high_tag_case::absent;
      com_ptr<ID3D12Resource> unsupported_depth;
      if(tag_fallback_test) {
        auto desc=selected->resource->GetDesc();desc.Format=DXGI_FORMAT_R16_FLOAT;
        desc.Flags=D3D12_RESOURCE_FLAG_NONE;
        const auto heap=heap_properties(D3D12_HEAP_TYPE_DEFAULT);
        checked(game->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_COMMON,
          nullptr,IID_PPV_ARGS(unsupported_depth.put())),"Create valid tagged identity with unsupported capture format");
      }
      void *function=cached_fg_options;
      if(!live_compat)
        require(get_function(1000,"slDLSSGSetOptions",function)==0 && function,"Fixture SDK did not return its real FG options entry point");
      require(function,"Live-compatible FG options were not cached before add-on discovery");
      set_options_t volatile set_options=reinterpret_cast<set_options_t>(function);
      // Continue ordinary game frames after SDK discovery. The returned callable
      // is the one the game invokes; the test never addresses an observer hook.
      for(unsigned i=0;i<4;++i) ngx_tick("FG-options-hook-discovery");
      abi_v2::viewport viewport{{nullptr,viewport_guid,1},live_compat?1u:0u};
      bool fg_enabled{},missing_fg_depth{};
      std::int32_t fg_tag_result{};
      const auto configure=[&](unsigned mode,unsigned generated) {
        fg_options options{{nullptr,{0xfac5f1cb,0x2dfd,0x4f36,{0xa1,0xe6,0x3a,0x9e,0x86,0x52,0x56,0xc5}},3},mode,generated};
        require(set_options(viewport,options)==0,"Observed FG options call changed the SDK result");
        fg_enabled=mode!=0;
      };

      struct publication { std::uint64_t generation{},sequence{},timestamp{},texture{},fence{};unsigned index{}; };
      struct mapping_guard {
        HANDLE handle{};wire::shared_state_t *state{};
        ~mapping_guard(){if(state)UnmapViewOfFile(state);if(handle)CloseHandle(handle);}
      } mapping;
      const auto mapping_name=std::wstring(wire::mapping_prefix)+std::to_wstring(GetCurrentProcessId());
      mapping.handle=OpenFileMappingW(FILE_MAP_ALL_ACCESS,FALSE,mapping_name.c_str());
      require(mapping.handle,"Frame-generation test cannot open the production exporter mapping");
      mapping.state=static_cast<wire::shared_state_t *>(MapViewOfFile(mapping.handle,FILE_MAP_ALL_ACCESS,0,0,sizeof(wire::shared_state_t)));
      require(mapping.state,"Frame-generation test cannot map actual exporter state");
      const auto set_foreground=reinterpret_cast<void (*)(HWND)>(GetProcAddress(GetModuleHandleW(L"SunshineSBSTest.addon64"),"SunshineSbsTestSetForeground"));
      require(set_foreground,"Frame-generation test needs the existing controlled foreground observer");
      set_foreground(window);
      InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&mapping.state->consumer_nonce),0x46475055424c4953);
      const auto read64=[](std::uint64_t &value){return std::uint64_t(InterlockedCompareExchange64(reinterpret_cast<volatile LONG64 *>(&value),0,0));};
      const auto published=[&]() {
        publication result;
        const auto before=InterlockedCompareExchange(reinterpret_cast<volatile LONG *>(&mapping.state->metadata_sequence),0,0);
        if(before&1) return result;
        const auto metadata=mapping.state->metadata;
        MemoryBarrier();
        if(before!=InterlockedCompareExchange(reinterpret_cast<volatile LONG *>(&mapping.state->metadata_sequence),0,0) ||
            !wire::valid_metadata(metadata)) return result;
        require(metadata.source_width==width && metadata.source_height==height && metadata.dxgi_format==10 &&
          metadata.color_transfer==wire::transfer::scrgb,"Actual FG export lost its HDR source geometry/transfer");
        result.generation=metadata.generation;result.fence=metadata.ready_fence_handle;
        for(unsigned i=0;i<wire::slot_count;++i) {
          auto &slot=mapping.state->slots[i];const auto control=read64(slot.control),sequence=read64(slot.sequence);
          if(wire::control_generation(control)==metadata.generation && wire::control_state(control)==wire::slot_state::ready && sequence>result.sequence)
            result={metadata.generation,sequence,read64(slot.qpc),metadata.texture_handles[i],metadata.ready_fence_handle,i};
        }
        return result;
      };
      const auto payload=[&](const publication &frame) {
        require(frame.sequence && frame.texture && frame.timestamp,"FG export has no completed frame to inspect");
        auto &slot=mapping.state->slots[frame.index];
        const auto ready_control=wire::slot_control(frame.generation,wire::slot_state::ready);
        const auto reading_control=wire::slot_control(frame.generation,wire::slot_state::reading);
        require(std::uint64_t(InterlockedCompareExchange64(reinterpret_cast<volatile LONG64 *>(&slot.control),reading_control,ready_control))==ready_control &&
          read64(slot.sequence)==frame.sequence,"FG export changed while acquiring its actual published slot");
        const auto unlock=[&]{InterlockedCompareExchange64(reinterpret_cast<volatile LONG64 *>(&slot.control),ready_control,reading_control);};
        try {
          com_ptr<ID3D12Fence> fence;com_ptr<ID3D12Resource> texture;
          checked(game->OpenSharedHandle(reinterpret_cast<HANDLE>(frame.fence),IID_PPV_ARGS(fence.put())),"Open actual FG exporter fence");
          require(fence->GetCompletedValue()>=frame.sequence && fence->GetCompletedValue()!=UINT64_MAX,"FG publication preceded GPU completion");
          checked(game->OpenSharedHandle(reinterpret_cast<HANDLE>(frame.texture),IID_PPV_ARGS(texture.put())),"Open actual FG exported texture");
          auto pixels=read(texture.p,D3D12_RESOURCE_STATE_COMMON);unlock();return pixels;
        } catch(...) {unlock();throw;}
      };
      const auto normal_depth_pixels=[&](const std::vector<std::uint8_t> &pixels,const char *phase) {
        require(pixels.size()==size_t(width)*height*2*8,"Normal Depth export has unexpected HDR pixel geometry");
        float low=1.f,high=0.f,eye_error{},gray_error{};
        for(unsigned y=height/8;y<height*7/8;y+=std::max(1u,height/180)) {
          for(unsigned x=width/8;x<width*7/8;x+=std::max(1u,width/320)) {
            const float left=channel(pixels,x,y,0),right=channel(pixels,width+x,y,0);
            require(std::isfinite(left) && std::isfinite(right),"Normal Depth export contains nonfinite depth");
            low=std::min(low,std::min(left,right));high=std::max(high,std::max(left,right));
            eye_error=std::max(eye_error,std::abs(left-right));
            for(unsigned eye=0;eye<2;++eye)for(unsigned component=1;component<3;++component) {
              const float value=channel(pixels,eye*width+x,y,component);
              require(std::isfinite(value),"Normal Depth export contains nonfinite color");
              gray_error=std::max(gray_error,std::abs(value-(eye?right:left)));
            }
          }
        }
        std::printf("MEASURE %s normal_depth_range=%.9g..%.9g gray_error=%.9g eye_error=%.9g\n",phase,low,high,gray_error,eye_error);
        require(low>=0.f && high<=1.f && high-low>.01f && gray_error<.002f && eye_error<.002f,
          "Actual Normal Depth export is flat, colored game mono, or inconsistent between eyes");
      };
      struct scale_display {unsigned basis{},state{};float value{};};
      const bool scale_ui_test=sunshine_camera_fixture::flag("SUNSHINE_FG_SCALE_UI_TEST");
      using query_scale_t=BOOL (*)(api::effect_runtime *,unsigned *,unsigned *,float *);
      const auto query_scale=reinterpret_cast<query_scale_t>(GetProcAddress(GetModuleHandleW(L"SunshineSBSTest.addon64"),"SunshineGame3DTestQueryScale"));
      require(!scale_ui_test || query_scale,"Scale UI regression setup requires the read-only production UI query export");
      const auto read_scale_ui=[&] {
        scale_display result;
        if(scale_ui_test)require(query_scale(observed.runtime,&result.basis,&result.state,&result.value),
          "Actual runtime has no published UI depth scale");
        return result;
      };
      const auto current_scale_ui=[&](unsigned expected_basis,const char *phase) {
        const auto display=read_scale_ui();
        if(scale_ui_test) {
          const float actual=scalar("Sunshine_CameraDepthScale");
          std::printf("MEASURE %s scale_UI_basis=%u state=%u value=%.9g shader_scale=%.9g\n",phase,display.basis,display.state,display.value,actual);
          require(ready() && display.basis==expected_basis && display.state==2 && std::isfinite(display.value) && display.value>0.f &&
              std::abs(display.value-actual)<=std::max(1e-5f,std::abs(actual)*1e-6f),
            "Active UI scale does not match its actual calculated/estimated shader coefficient");
        }
        return display;
      };
      const auto retained_scale_ui=[&](const scale_display &before,const char *phase) {
        if(!scale_ui_test)return;
        const auto current=read_scale_ui();
        std::printf("MEASURE %s retained_scale_UI_basis=%u state=%u value=%.9g\n",phase,current.basis,current.state,current.value);
        require(current.basis==before.basis && current.state==before.state && current.value==before.value,
          "Reusing real FG depth changed its displayed calibration scale");
      };
      configure(0,0);
      ngx_settle("FG-export-warmup");
      for(unsigned i=0;i<20 && !published().sequence;++i) ngx_tick("FG-consumer-handshake");
      require(published().sequence,"FG consumer did not receive a real exported frame");

      // Change actual game color without touching a shader output or exporter
      // slot. The row-constant green channel is a current-color oracle under
      // horizontal stereo warping; blue retains its stereo texture structure.
      com_ptr<ID3D12Resource> alternate_upload;
      auto alternate_bytes=source_bytes;
      for(size_t i=0;i<alternate_bytes.size();i+=8) {
        const std::uint16_t red=0x4200,green=0x3400;
        std::memcpy(alternate_bytes.data()+i,&red,2);std::memcpy(alternate_bytes.data()+i+2,&green,2);
      }
      D3D12_PLACED_SUBRESOURCE_FOOTPRINT alternate_footprint{};
      fill_upload(alternate_upload,backbuffers[0]->GetDesc(),alternate_bytes.data(),alternate_footprint);
      require(alternate_footprint.Offset==source_footprint.Offset && alternate_footprint.Footprint.RowPitch==source_footprint.Footprint.RowPitch,
        "FG alternate color upload changed its copy layout");
      const auto change_color=[&]{std::swap(source_upload.p,alternate_upload.p);source_bytes.swap(alternate_bytes);};
      const auto check_jitter=[&](const sunshine_scene_depth::jitter_offset &expected) {
        const auto &actual=captured.provided.jitter;
        require(actual.supplied==expected.supplied && actual.x==expected.x && actual.y==expected.y &&
          actual.width==expected.width && actual.height==expected.height,
          "Captured real depth lost its own render-pixel jitter metadata");
        float offset[2]{};
        observed.runtime->get_uniform_value_float(uniform("Sunshine_DepthJitter"),offset,2);
        // Here the render domain equals the selected depth crop. One render
        // pixel therefore remains one allocation pixel despite nonzero crop
        // origin and a different final-color resolution.
        const float expected_offset[]{expected.supplied?expected.x/selected->width:0.f,
          expected.supplied?expected.y/selected->height:0.f};
        for(unsigned i=0;i<2;++i)require(std::abs(offset[i]-expected_offset[i])<1e-8f,
          "Shader jitter does not belong to the captured depth (or was normalized by output/crop size)");
      };
      const auto check_reused_depth=[&](const sunshine_depth::frame_depth &previous) {
        require(captured_ready && ready() && captured.reused_depth && captured_render==observed.renders &&
          captured.frame_index>previous.frame_index && captured.source_id==previous.source_id &&
          captured.resource.handle==previous.resource.handle && captured.provided.sequence==previous.provided.sequence &&
          captured.provided.tick==previous.provided.tick && captured.projection.A==previous.projection.A &&
          captured.projection.B==previous.projection.B && captured.projection.raw_scale==previous.projection.raw_scale &&
          captured.projection.raw_bias==previous.projection.raw_bias,
          "FG reuse did not expose the previous completed real depth and calibration for the current presentation");
        check_jitter(previous.provided.jitter);
        require(scalar("Sunshine_CameraStrengthBlend")==1.f,
          "Reusing real FG depth restarted the stereo strength ramp");
      };
      const auto check_fresh_stereo=[&](const publication &previous,const sunshine_depth::frame_depth &depth,
          const std::vector<std::uint8_t> &depth_pixels,const char *phase,bool depth_view=false) {
        check_reused_depth(depth);
        const auto current=published();
        require(current.generation==previous.generation && current.sequence>previous.sequence && current.timestamp>previous.timestamp,
          "FG reuse held the old stereo publication instead of publishing the current presentation");
        const auto pixels=payload(current);
        require(pixels==read(exported.p),"FG reuse did not export the actual current shader result");
        require(read(reinterpret_cast<ID3D12Resource *>(captured.resource.handle))==depth_pixels,
          "FG reuse read pending/poisoned depth instead of the previous completed depth texture");
        if(depth_view)normal_depth_pixels(pixels,phase);
        else {
          float green_error{},stereo_difference{};
          for(unsigned y=height/8;y<height*7/8;y+=std::max(1u,height/90)) {
            std::uint16_t expected_half{};
            std::memcpy(&expected_half,source_bytes.data()+size_t(y)*width*8+2,2);
            const float expected=half_float(expected_half);
            for(unsigned x=width/8;x<width*7/8;x+=std::max(1u,width/160)) {
              for(unsigned eye=0;eye<2;++eye)
                green_error=std::max(green_error,std::abs(channel(pixels,eye*width+x,y,1)-expected));
              stereo_difference=std::max(stereo_difference,std::abs(channel(pixels,x,y,2)-channel(pixels,width+x,y,2)));
            }
          }
          std::printf("MEASURE %s current_color_error=%.9g stereo_difference=%.9g depth_tick=%llu sequence=%llu\n",
            phase,green_error,stereo_difference,static_cast<unsigned long long>(captured.provided.tick),
            static_cast<unsigned long long>(current.sequence));
          require(green_error<.003f && stereo_difference>.01f,
            "FG output reused old color or duplicated mono instead of rendering new-color stereo");
        }
        return current;
      };
      render_tracked_depth=[&]{if(emit)record_ngx_frame();};
      const auto current_mono=[&](const publication &previous,const char *phase) {
        ngx_tick(phase);const auto current=published();
        require(current.generation==previous.generation && current.sequence>previous.sequence,"A real missing-depth/off-FG frame incorrectly retained historical stereo");
        require(!ready() && !captured_ready,"A real missing-depth frame reused historical scene depth");
        require(payload(current)==check_current_mono(),"Published fallback does not contain the actual current-color mono output");
        return current;
      };

      // FG source intent takes priority before any SL depth arrives. Even fresh
      // valid NGX evaluations cannot supply pixels or camera for this interval.
      configure(1,1);
      for(unsigned i=0;i<3;++i) {
        change_color();current_mono(published(),"FG-waiting-for-first-SL-with-fresh-NGX");
        require(captured.provided.provider==sunshine_scene_depth::provider_kind::streamline &&
          captured.provided.frame_generation_input && captured.provided.viewport==viewport.value &&
          !captured.projection.supplied,"Waiting FG retained NGX metadata or an unrelated camera");
      }
      emit=false;change_color();current_mono(published(),"FG-waiting-for-first-SL-with-NGX-gap");
      configure(0,0);emit=true;ngx_settle("FG-off-before-first-SL-returns-NGX");
      require(captured.provided.provider==sunshine_scene_depth::provider_kind::ngx,
        "FG Off before first SL capture did not restore ordinary NGX depth");
      configure(1,1);
      std::puts("PASS FG waits for SL depth despite fresh NGX frames, and confirmed Off restores NGX");

      const auto new_token=reinterpret_cast<abi_v2::get_new_frame_token>(GetProcAddress(interposer,"slGetNewFrameToken"));
      const auto set_constants=reinterpret_cast<abi_v2::set_constants>(GetProcAddress(interposer,"slSetConstants"));
      const auto set_tag=reinterpret_cast<abi_v2::set_tag_for_frame>(GetProcAddress(interposer,"slSetTagForFrame"));
      const auto set_global_tag=reinterpret_cast<abi_v2::set_tag>(GetProcAddress(interposer,"slSetTag"));
      const auto evaluate_sr=reinterpret_cast<abi_v2::evaluate_feature>(GetProcAddress(interposer,"slEvaluateFeature"));
      const auto set_tag_result=reinterpret_cast<void (*)(std::int32_t)>(GetProcAddress(interposer,"SunshineFixtureSetTagResult"));
      require(new_token && set_constants && set_tag && set_global_tag && evaluate_sr && set_tag_result,"SLFG fixture metadata exports are missing");
      struct precision_info {base_structure base;std::uint32_t formula{};float bias{},scale{};};
      static_assert(sizeof(precision_info)==48);
      precision_info precision{{nullptr,{0x98f6e9ba,0x8d16,0x4831,{0xa8,0x02,0x4d,0x3b,0x52,0xff,0x26,0xbf}},1},1,.01f,2.f};
      if(sunshine_camera_fixture::flag("SUNSHINE_FG_IDENTITY_PRECISION_TEST")) {precision.bias=0;precision.scale=1;}
      initialize_camera();
      com_ptr<ID3D12Resource> boundary_witness;
      const auto witness_heap=heap_properties(D3D12_HEAP_TYPE_DEFAULT);
      const auto witness_desc=selected->resource->GetDesc();
      checked(game->CreateCommittedResource(&witness_heap,D3D12_HEAP_FLAG_NONE,&witness_desc,
        D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(boundary_witness.put())),"Create independent FG tag-boundary witness");
      // Poison is an ordinary GPU buffer-to-texture copy, independent of clear
      // intrinsics or descriptor state. The compute scene producer is unchanged.
      com_ptr<ID3D12Resource> poison_upload;
      D3D12_PLACED_SUBRESOURCE_FOOTPRINT poison_footprint{};
      const std::vector<float> poison_values(size_t(selected->width)*selected->height,.75f);
      fill_upload(poison_upload,witness_desc,poison_values.data(),poison_footprint);
      auto *const poison_resource=selected->resource.p;
      std::uint32_t fg_frame{};
      float fg_jitter_x{},fg_jitter_y{};
      std::uint64_t fg_recording_at_start{},fg_recording_at_tag{};
      alignas(void *) unsigned char opaque_token[16]{};
      record_fg_tags=[&] {
        fg_recording_at_start=game_native_command;
        if(!fg_enabled)return;
        require(selected->resource.p==poison_resource,"FG fixture rotated the fixed volatile-source scenario");
        abi_v2::frame_token *token{};const auto index=++fg_frame;
        if(live_compat) token=reinterpret_cast<abi_v2::frame_token *>(opaque_token);
        else require(new_token(token,&index)==0 && token,"SLFG fixture failed to obtain its real explicit frame token");
        abi_v2::constants constants{};constants.base={nullptr,constants_guid,1};
        constants.common.camera_view_to_clip=camera.projection;constants.common.clip_to_camera_view=camera.inverse_projection;
        constants.common.camera_near=camera.near_plane;constants.common.camera_far=camera.far_plane;
        constants.common.camera_fov=camera.fov;constants.common.camera_aspect=camera.aspect;
        constants.common.jitter_offset[0]=fg_jitter_x;constants.common.jitter_offset[1]=fg_jitter_y;
        constants.common.camera_right[0]=constants.common.camera_up[1]=constants.common.camera_forward[2]=1;
        constants.depth_inverted=1;
        require(set_constants(constants,*token,viewport)==0,"SLFG constants original result changed");
        abi_v2::resource resource{};resource.base={nullptr,resource_guid,1};
        // Expedition's tags use eUnknown; the shared owner must establish the
        // actual texture interface/device rather than infer it from this enum.
        resource.type=8;
        if(live_compat) {resource.base={};resource.type=0;}
        resource.native=selected->resource.p;resource.state=unsigned(selected->state);
        resource.width=selected->width;resource.height=selected->height;resource.native_format=DXGI_FORMAT_R32_FLOAT;
        resource.mip_levels=resource.array_layers=1;
        // Match the live zero-initialized wrapper: the shared owner must get
        // dimensions from the native object and state from observed barriers.
        if(live_compat) {resource.state=0;resource.width=resource.height=0;}
        abi_v2::resource_tag tag{{&precision,tag_guid,1},missing_fg_depth?nullptr:&resource,0,0,{top,left,active_width,active_height}};
        auto high_resource=resource;
        high_resource.native=high_tag_mode==high_tag_case::usable?selected->resource.p:unsupported_depth.p;
        high_resource.native_format=high_tag_mode==high_tag_case::usable?DXGI_FORMAT_R32_FLOAT:DXGI_FORMAT_R16_FLOAT;
        high_resource.state=high_tag_mode==high_tag_case::usable?resource.state:0;
        abi_v2::resource_tag tags[]{tag,{{&precision,tag_guid,1},&high_resource,48,0,{top,left,active_width,active_height}}};
        if(high_tag_mode==high_tag_case::neither_usable) tags[0].resource_ptr=&high_resource;
        const auto tag_count=high_tag_mode==high_tag_case::absent?1u:2u;
        set_tag_result(fg_tag_result);
        // Independent GPU witness on the fixture's own producer list. It proves
        // what pixels existed at the public SDK boundary without waiting for GPU
        // work or invoking any add-on capture/readiness seam.
        transition(commands.p,selected->resource.p,selected->state,D3D12_RESOURCE_STATE_COPY_SOURCE);
        commands->CopyResource(boundary_witness.p,selected->resource.p);
        if(contiguous_copy) {
          selected->state=D3D12_RESOURCE_STATE_COPY_SOURCE;
          resource.state=live_compat ? 0u : unsigned(selected->state);
        } else transition(commands.p,selected->resource.p,D3D12_RESOURCE_STATE_COPY_SOURCE,selected->state);
        fg_recording_at_tag=game_native_command;
        if(cross_queue && (producer_gated || fg_recording_at_tag!=producer_recording_native))
          std::printf("MEASURE SLFG native recording expected_producer=0x%llx start=0x%llx at_tag=0x%llx gated=%u\n",
            static_cast<unsigned long long>(producer_recording_native),static_cast<unsigned long long>(fg_recording_at_start),
            static_cast<unsigned long long>(fg_recording_at_tag),unsigned(producer_gated));
        require(!cross_queue || fg_recording_at_tag==producer_recording_native,
          "Fixture SLFG tag targeted a different native command list than its actual depth producer");
        if(!(producer_oracle_only && producer_gated)) {
          const auto tag_result=live_compat ? set_global_tag(viewport,tags,tag_count,reinterpret_cast<void *>(game_native_command)) :
            set_tag(*token,viewport,tags,tag_count,reinterpret_cast<void *>(game_native_command));
          require(tag_result==fg_tag_result,
            "SLFG tag observation changed the actual SDK success/failure result");
          const base_structure *sr_inputs[]{&viewport.base};
          require(evaluate_sr(0,*token,sr_inputs,1,reinterpret_cast<void *>(game_native_command))==0,
            "Same-viewport SR observation changed the original SDK result");
        }
        // OnlyValidNow ends at the original tag call. Destroy the input pixels
        // immediately: a later EOF recopy cannot satisfy the following oracle.
        transition(commands.p,selected->resource.p,selected->state,D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION poison_from{},poison_to{};
        poison_from.pResource=poison_upload.p;poison_from.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        poison_from.PlacedFootprint=poison_footprint;
        poison_to.pResource=selected->resource.p;poison_to.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        commands->CopyTextureRegion(&poison_to,0,0,0,&poison_from,nullptr);
        transition(commands.p,selected->resource.p,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        selected->state=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
      };
      const auto sl_current=[&] {
        return captured_ready && captured_render==observed.renders && captured.provided.frame_generation_input &&
          captured.provided.provider==sunshine_scene_depth::provider_kind::streamline && captured.source_id==((1ull<<63)|viewport.value) &&
          captured.projection.supplied && captured.source_resource.handle==native(*selected) &&
          (live_compat ? !captured.provided.source_frame_explicit && !captured.provided.source_frame_has_numeric :
            captured.provided.source_frame_explicit && captured.provided.source_frame_numeric==fg_frame);
      };
      const auto settle_fg=[&](const char *phase) {
        unsigned stable{};const auto deadline=GetTickCount64()+15000;
        do {ngx_tick(phase);stable=sl_current() && ready() && scalar("Sunshine_CameraStrengthBlend")==1.f?stable+1:0;}
        while(stable<6 && GetTickCount64()<deadline);
        if(stable!=6) {
          report_debug_messages();
          const auto now=GetTickCount64();
          const auto &metadata=captured.provided;
          std::printf("MEASURE failed %s current=%u capture_ready=%u shader_ready=%u now=%llu metadata_tick=%llu epoch=%llu sequence=%llu source=%llu viewport=%u frame=%llu expected_frame=%u explicit=%u FG=%u A=%.17g B=%.17g raw_scale=%.17g raw_bias=%.17g display=0x%llx original=0x%llx\n",
            phase,unsigned(sl_current()),unsigned(captured_ready),unsigned(ready()),
            static_cast<unsigned long long>(now),static_cast<unsigned long long>(metadata.tick),
            static_cast<unsigned long long>(metadata.epoch),static_cast<unsigned long long>(metadata.sequence),
            static_cast<unsigned long long>(metadata.source_id),metadata.viewport,
            static_cast<unsigned long long>(metadata.source_frame_numeric),fg_frame,unsigned(metadata.source_frame_explicit),
            unsigned(metadata.frame_generation_input),captured.projection.A,captured.projection.B,
            captured.projection.raw_scale,captured.projection.raw_bias,
            static_cast<unsigned long long>(captured.resource.handle),static_cast<unsigned long long>(native(*selected)));
          using center_query_t=bool (*)(api::effect_runtime *,sunshine_streamline::provider::center_sample *);
          const auto query_center=reinterpret_cast<center_query_t>(GetProcAddress(GetModuleHandleW(L"SunshineSBSTest.addon64"),"SunshineStreamlineTestCenter"));
          sunshine_streamline::provider::center_sample measured;
          const bool have_center=query_center && query_center(observed.runtime,&measured);
          std::printf("MEASURE failed FG latest_center export=%u available=%u id=%llu tick=%llu now=%llu epoch=%llu viewport=%u sequence=%llu source=%llu projection=%u A=%.17g B=%.17g raw_scale=%.17g raw_bias=%.17g central_raw=",
            unsigned(query_center!=nullptr),unsigned(have_center),static_cast<unsigned long long>(measured.id),
            static_cast<unsigned long long>(measured.tick),static_cast<unsigned long long>(now),
            static_cast<unsigned long long>(measured.projection.epoch),measured.projection.viewport,
            static_cast<unsigned long long>(measured.metadata.sequence),static_cast<unsigned long long>(measured.metadata.source_id),
            unsigned(measured.projection.supplied),measured.projection.A,measured.projection.B,
            measured.projection.raw_scale,measured.projection.raw_bias);
          if (measured.width && measured.height && std::uint64_t(measured.width)*measured.height<=measured.raw.size()) {
            const auto columns=std::min(measured.width,4u),rows=std::min(measured.height,4u);
            const auto left_sample=(measured.width-columns)/2,top_sample=(measured.height-rows)/2;
            for(unsigned y=0;y<rows;++y)for(unsigned x=0;x<columns;++x)
              std::printf("%.9g%s",measured.raw[(top_sample+y)*measured.width+left_sample+x],x+1==columns && y+1==rows?"\n":",");
          } else std::puts("unavailable");
          const auto original_bytes=read(selected->resource.p,selected->state);
          const auto witness_bytes=read(boundary_witness.p,D3D12_RESOURCE_STATE_COPY_DEST);
          const auto original_offset=(size_t(top+active_height/2)*selected->width+left+active_width/2)*4;
          float original_center{},witness_center{};std::memcpy(&original_center,original_bytes.data()+original_offset,4);
          std::memcpy(&witness_center,witness_bytes.data()+original_offset,4);
          std::printf("MEASURE failed FG producer witness_center=%.9g expected=%.9g native_recording_start=0x%llx native_recording_tag=0x%llx\n",
            witness_center,center_raw,static_cast<unsigned long long>(fg_recording_at_start),static_cast<unsigned long long>(fg_recording_at_tag));
          using snapshot_query_t=bool (*)(api::effect_runtime *,std::uint64_t *,bool *,bool *);
          const auto query_snapshot=reinterpret_cast<snapshot_query_t>(GetProcAddress(GetModuleHandleW(L"SunshineSBSTest.addon64"),"SunshineStreamlineTestSnapshot"));
          std::uint64_t snapshot_texture{};bool forced{},shared{};
          const bool have_snapshot=query_snapshot && query_snapshot(observed.runtime,&snapshot_texture,&forced,&shared);
          if(have_snapshot && snapshot_texture) {
            const auto snapshot_bytes=read(reinterpret_cast<ID3D12Resource *>(snapshot_texture));
            float snapshot_center{};std::memcpy(&snapshot_center,snapshot_bytes.data()+original_offset,4);
            std::printf("MEASURE failed FG owned snapshot=0x%llx center=%.9g forced=%u shared=%u\n",
              static_cast<unsigned long long>(snapshot_texture),snapshot_center,unsigned(forced),unsigned(shared));
          } else std::printf("MEASURE failed FG owned snapshot export=%u available=%u texture=0x%llx forced=%u shared=%u\n",
            unsigned(query_snapshot!=nullptr),unsigned(have_snapshot),static_cast<unsigned long long>(snapshot_texture),unsigned(forced),unsigned(shared));
          if(captured_ready && captured.resource.handle) {
            const auto display_bytes=read(reinterpret_cast<ID3D12Resource *>(captured.resource.handle));
            const auto display_offset=(size_t(captured.y+captured.active_height/2)*captured.width+captured.x+captured.active_width/2)*4;
            float display_center{};std::memcpy(&display_center,display_bytes.data()+display_offset,4);
            std::printf("MEASURE failed FG actual center display=%.9g original=%.9g expected_display=%.9g poison=.75 allocation=%ux%u rect=%u,%u,%u,%u\n",
              display_center,original_center,center_raw,captured.width,captured.height,captured.x,captured.y,captured.active_width,captured.active_height);
          } else std::printf("MEASURE failed FG actual center display=unavailable original=%.9g expected_display=%.9g poison=.75\n",original_center,center_raw);
        }
        require(stable==6,"SLFG OnlyValidNow source did not reach current projection-based stereo");
      };
      settle_fg("SLFG-OnlyValidNow-precision-source");
      if(tag_fallback_test) {
        const auto check_kind=[&](sunshine_scene_depth::resource_kind kind,const char *message) {
          require(sl_current() && ready() && captured.provided.resource.kind==kind,message);
          require(selected_binding().handle==captured.shader_resource.handle,
            "Selected SL tag did not reach the actual shader binding");
          require(read(reinterpret_cast<ID3D12Resource *>(captured.resource.handle))==
            read(boundary_witness.p,D3D12_RESOURCE_STATE_COPY_DEST),
            "Selected SL tag did not preserve the actual current depth pixels");
        };
        check_kind(sunshine_scene_depth::resource_kind::raw_depth,"Unsupported tag48 blocked usable tag0 depth");
        high_tag_mode=high_tag_case::usable;
        settle_fg("SL-usable-tag48-priority");
        check_kind(sunshine_scene_depth::resource_kind::display_depth,"Usable tag48 did not retain priority over tag0");
        high_tag_mode=high_tag_case::unsupported;
        settle_fg("SL-tag48-failure-recovery-to-tag0");
        check_kind(sunshine_scene_depth::resource_kind::raw_depth,"Tag48 failure did not recover to tag0");
        high_tag_mode=high_tag_case::neither_usable;
        change_color();current_mono(published(),"SL-both-tags-unsupported");
        require(captured.provided.provider==sunshine_scene_depth::provider_kind::streamline,
          "Unsupported SL pixels switched source authority to another provider");
        high_tag_mode=high_tag_case::unsupported;
        settle_fg("SL-tag0-recovery-after-both-unsupported");
        check_kind(sunshine_scene_depth::resource_kind::raw_depth,"Tag0 did not recover after both captures failed");
        std::puts("PASS real SL hooks prefer usable tag48, fall back to tag0 on capture failure, and retain SL ownership with current-color mono when both fail");
      }
      require(captured.projection.A==0 && captured.projection.B==.0625 &&
        captured.projection.raw_scale==precision.scale && captured.projection.raw_bias==precision.bias,
        "SLFG PrecisionInfo changed the actual camera projection or lost its independent encoding transform");
      const auto &depth_projection=captured.projection;
      const double center_inverse_depth=(double(center_raw)*depth_projection.raw_scale+depth_projection.raw_bias-
        depth_projection.A)/depth_projection.B;
      const float expected_camera_scale=float(1./center_inverse_depth);
      require(std::isfinite(expected_camera_scale) && expected_camera_scale>0.f,
        "SLFG fixture has no valid physical inverse-depth center for scene calibration");
      const bool legacy_matrix_gain=sunshine_camera_fixture::flag("SUNSHINE_EXPECT_LEGACY_MATRIX_GAIN");
      const float expected_gain=legacy_matrix_gain?16.f:expected_camera_scale;
      require(std::abs(scalar("Sunshine_CameraDepthScale")-expected_gain)<=expected_gain*1e-5f,
        "SLFG scene gain does not match the explicitly selected control/treatment normalization policy");
      std::printf("MEASURE SLFG gain_oracle=%s physical_q=%.9g expected_gain=%.9g actual_gain=%.9g\n",
        legacy_matrix_gain?"frozen-legacy-matrix":"scene-physical-inverse-depth",center_inverse_depth,expected_gain,
        scalar("Sunshine_CameraDepthScale"));
      current_scale_ui(1,"SLFG-ready-calculated-scale");
      float projection_uniform[2]{};observed.runtime->get_uniform_value_float(uniform("Sunshine_CameraProjection"),projection_uniform,2);
      require(std::abs(projection_uniform[0]+precision.bias/precision.scale)<1e-7f && projection_uniform[1]==16.f*precision.scale,
        "SLFG shader reconstruction ignored nonidentity PrecisionInfo");
      const auto preserved=read(reinterpret_cast<ID3D12Resource *>(captured.resource.handle));
      const auto destroyed=read(selected->resource.p,selected->state);
      const auto witness=read(boundary_witness.p,D3D12_RESOURCE_STATE_COPY_DEST);
      const auto center=(size_t(top+active_height/2)*selected->width+left+active_width/2)*4;
      float retained_center{},destroyed_center{},witness_center{};std::memcpy(&retained_center,preserved.data()+center,4);std::memcpy(&destroyed_center,destroyed.data()+center,4);
      std::memcpy(&witness_center,witness.data()+center,4);
      require(witness_center==center_raw,"Fixture producer did not supply the expected real GPU depth at the SLFG tag boundary");
      require(retained_center==center_raw && destroyed_center==.75f,"SLFG OnlyValidNow failed to snapshot depth before the game destroyed it");
      const auto prepared=read(linear_depth.p);const auto prepared_desc=linear_depth->GetDesc();
      std::uint16_t center_half{};
      std::memcpy(&center_half,prepared.data()+(size_t(prepared_desc.Height/2)*prepared_desc.Width+prepared_desc.Width/2)*4+2,2);
      const float expected_prepared=1.f/(1.f+scalar("Sunshine_CameraDepthScale")*(center_raw-projection_uniform[0])*projection_uniform[1]);
      require(std::abs(half_float(center_half)-expected_prepared)<.002f,"Actual SLFG prepared depth does not use current pixels and transformed projection coefficients");
      const auto source_stereo=read(exported.p);
      sunshine_parity::write_bytes(runtime_directory/"fg-source.sbs",source_stereo.data(),source_stereo.size());
      std::puts("PASS actual SLFG tags pass native texture validation, apply PrecisionInfo and snapshot OnlyValidNow depth before poison; same-viewport SR evaluation preserves FG ownership");
      if(live_compat) std::puts("PASS pre-discovery cached FG options, zero-base Resource and global OnlyValidNow tags with untracked opaque token reach actual projection stereo");

      fg_jitter_x=.375f;fg_jitter_y=-.25f;
      settle_fg("SLFG-fresh-depth-with-jitter");
      check_jitter({fg_jitter_x,fg_jitter_y,active_width,active_height,true});

      if(late_source_test) {
        require(cross_queue && complete_producer && retire_producer_recording && !producer_oracle_only,
          "Late 4x FG regression requires the real separate producer queue");
        configure(1,3);settle_fg("SLFG-4x-late-initial-source");
        const auto real_pixels=read(reinterpret_cast<ID3D12Resource *>(captured.resource.handle));
        ngx_tick("SLFG-4x-late-refresh-real-source");
        const auto real_depth=captured;const auto real_publication=published();
        const auto real_scale=current_scale_ui(1,"SLFG-4x-late-initial-scale");
        com_ptr<ID3D12Fence> gate;
        checked(game->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(gate.put())),
          "Create delayed 4x real-depth producer gate");
        HANDLE stop_watchdog=CreateEventW(nullptr,TRUE,FALSE,nullptr);
        require(stop_watchdog,"Create delayed 4x producer cleanup event");
        std::atomic<bool> emergency_release{false};
        std::thread watchdog([&] {
          if(WaitForSingleObject(stop_watchdog,2000)!=WAIT_OBJECT_0) {
            emergency_release=true;gate->Signal(1);
          }
        });
        const auto release_producer=[&] {
          gate->Signal(1);SetEvent(stop_watchdog);watchdog.join();CloseHandle(stop_watchdog);
          independent_generated_present=false;producer_gated=false;emit=true;
          checked(producer_completion->SetEventOnCompletion(producer_fence_value,completion_event),
            "Observe delayed real producer completion during test cleanup");
          require(WaitForSingleObject(completion_event,3000)==WAIT_OBJECT_0,
            "Delayed real producer did not complete after its test gate was released");
        };
        try {
          // FG's generated color is already available independently of the next
          // real depth. Unlike the ordinary fixture, this presentation must not
          // wait for that unrelated future producer. The add-on receives actual
          // pending SDK capture metadata and must not add a reverse queue wait.
          independent_generated_present=true;producer_gated=true;
          checked(producer_queue->Wait(gate.p,1),"Delay the next real FG producer across generated presentations");
          change_color();ngx_tick("SLFG-4x-late-stage-pending-real-source");
          require(!emergency_release && producer_completion->GetCompletedValue()<producer_fence_value,
            "An independent generated presentation waited for the future real-depth producer");
          check_reused_depth(real_depth);
          emit=false;
          const auto age=GetTickCount64()-real_depth.provided.tick;
          if(age<120)Sleep(DWORD(120-age));
          auto previous=published();
          for(unsigned i=0;i<3;++i) {
            change_color();ngx_tick("SLFG-4x-late-generated-color");
            const auto source_age=GetTickCount64()-real_depth.provided.tick;
            const auto output=published();
            std::printf("MEASURE 4x_late generated=%u depth_age_ms=%llu captured_ready=%u shader_ready=%u reused=%u source_sequence=%llu output_sequence=%llu producer_pending=%u\n",
              i+1,static_cast<unsigned long long>(source_age),unsigned(captured_ready),unsigned(ready()),unsigned(captured.reused_depth),
              static_cast<unsigned long long>(captured.provided.sequence),static_cast<unsigned long long>(output.sequence),
              unsigned(producer_completion->GetCompletedValue()<producer_fence_value));
            require(source_age>=120 && source_age<250,
              "Late 4x test presentation did not fit the intended 120-250ms source-age interval");
            require(!emergency_release && producer_completion->GetCompletedValue()<producer_fence_value,
              "Generated FG color depended on the unfinished real-depth queue");
            check_reused_depth(real_depth);
            require(output.generation==previous.generation && output.sequence>previous.sequence && output.timestamp>previous.timestamp,
              "Late 4x generated presentation did not publish new color");
            retained_scale_ui(real_scale,"SLFG-4x-late-generated-scale");
            previous=output;
          }
          check_fresh_stereo(real_publication,real_depth,real_pixels,"SLFG-4x-late-final-pixels");
          // Neither generated presentations nor a still-pending nomination may
          // renew the timestamp of the old real capture or extend its deadline.
          const auto age_after_pixels=GetTickCount64()-real_depth.provided.tick;
          if(age_after_pixels<270)Sleep(DWORD(270-age_after_pixels));
          change_color();current_mono(published(),"SLFG-4x-late-source-expired");
          require(!emergency_release && producer_completion->GetCompletedValue()<producer_fence_value,
            "The delayed producer completed unexpectedly before the bounded-expiry check");
        } catch(...) {
          release_producer();throw;
        }
        release_producer();
        require(!emergency_release,"The add-on blocked independent FG presentations until watchdog release");
        settle_fg("SLFG-4x-late-fresh-real-recovery");
        set_foreground(nullptr);record_fg_tags={};
        std::puts("PASS 4x generated color remains fresh with unchanged 120-250ms real-depth identity/calibration while the next producer is pending; depth expires after250ms and fresh capture recovers without an add-on wait");
        return;
      }

      if(sunshine_camera_fixture::flag("SUNSHINE_FG_SUSTAINED_PRODUCER_TEST")) {
        require(cross_queue && complete_producer && retire_producer_recording && !producer_oracle_only,
          "Sustained FG pipeline requires a separately queued, genuinely retired producer");
        const auto initial_pattern=selected->pattern;
        // Compute the full independent pixel oracles before starting the age-
        // bounded stream. Repeating millions of divisions and scalar channel
        // checks between 4K presents would itself age the prior capture out.
        std::array<std::vector<float>,2> expected_depth;
        std::array<std::vector<std::uint8_t>,2> expected_mono;
        require(color==2,"Sustained FG exact color oracle requires the scRGB fixture");
        std::uint16_t first_color_green{};
        std::memcpy(&first_color_green,source_bytes.data()+2,2);
        for(unsigned pattern=0;pattern<2;++pattern) {
          auto &depth=expected_depth[pattern];depth.resize(size_t(selected->width)*selected->height,.9375f);
          for(unsigned y=top;y<top+active_height;++y)for(unsigned x=left;x<left+active_width;++x) {
            const auto cx=std::min(31u,unsigned((double(x-left)+.5)*32/active_width));
            const auto cy=std::min(17u,unsigned((double(y-top)+.5)*18/active_height));
            float value=.0078125f*float((pattern?31-cx:cx)/8+1);
            if(cx>=14 && cx<18 && cy>=7 && cy<11)value=center_raw;
            depth[size_t(y)*selected->width+x]=value;
          }
          const auto &source=pattern?alternate_bytes:source_bytes;
          auto &mono=expected_mono[pattern];mono.resize(size_t(width)*height*2*8);
          for(unsigned y=0;y<height;++y)for(unsigned eye=0;eye<2;++eye)
            std::memcpy(mono.data()+(size_t(y)*width*2+eye*width)*8,source.data()+size_t(y)*width*8,size_t(width)*8);
        }
        const auto exact_current_mono=[&](const publication &frame) {
          std::uint16_t green{};std::memcpy(&green,source_bytes.data()+2,2);
          require(payload(frame)==expected_mono[green!=first_color_green],
            "Sustained FG mono fallback differs from the actual current HDR source in either eye");
        };
        const auto current_mono_fast=[&](const publication &previous,const char *phase) {
          ngx_tick(phase);const auto output=published();
          require(!captured_ready && !ready() && output.generation==previous.generation && output.sequence>previous.sequence,
            "Sustained FG failure did not publish a new current-color mono frame");
          exact_current_mono(output);
        };
        settle_fg("SLFG-sustained-initial-real-depth");
        const auto initial_depth=captured;
        const auto initial_scale=current_scale_ui(1,"SLFG-sustained-initial-scale");
        auto previous_publication=published();
        auto previous_real_sequence=captured.provided.sequence;
        constexpr unsigned frame_count=16;
        unsigned pending_frames{},ready_frames{},mono_frames{},refreshed_frames{},stale_frames{},wrong_depth_frames{};
        const auto started=GetTickCount64();
        auto previous_present_end=started;
        const auto pending_present=[&](const char *phase) {
          com_ptr<ID3D12Fence> gate;
          checked(game->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(gate.put())),
            "Create sustained FG producer gate");
          HANDLE observed_event=CreateEventW(nullptr,TRUE,FALSE,nullptr);
          require(observed_event,"Create sustained FG observation event");
          bool observed_effect{},observed_pending{};
          HRESULT release_result=E_FAIL;
          std::atomic<bool> emergency_release{false};
          ngx_frame_observer=[&] {
            if(observed_effect)return;
            observed_effect=true;
            const auto completed=producer_completion->GetCompletedValue();
            observed_pending=completed!=UINT64_MAX && completed<producer_fence_value;
            release_result=gate->Signal(1);SetEvent(observed_event);
          };
          std::thread watchdog([&] {
            if(WaitForSingleObject(observed_event,2000)!=WAIT_OBJECT_0) {
              emergency_release=true;gate->Signal(1);
            }
          });
          producer_gated=true;change_color();
          try {
            checked(producer_queue->Wait(gate.p,1),"Keep latest FG producer pending until effect acquisition");
            ngx_tick(phase);
          } catch(...) {
            gate->Signal(1);SetEvent(observed_event);watchdog.join();
            ngx_frame_observer={};producer_gated=false;CloseHandle(observed_event);throw;
          }
          watchdog.join();ngx_frame_observer={};producer_gated=false;CloseHandle(observed_event);
          require(!emergency_release && observed_effect && observed_pending && SUCCEEDED(release_result),
            "Sustained FG producer was not pending or acquisition blocked the CPU");
        };
        for(unsigned i=0;i<frame_count;++i) {
          const auto frame_started=GetTickCount64();
          const auto interframe_gap=frame_started-previous_present_end;
          // step() waits for the GAME's main queue after effects. Its existing
          // producer wait therefore proves N is completed before N+1 is tagged.
          // There is deliberately no completed-source warmup between frames:
          // nomination/allocation of N+1 must not discard the unconsumed N copy.
          const auto completed_before_tag=producer_completion->GetCompletedValue();
          require(completed_before_tag!=UINT64_MAX && completed_before_tag>=producer_fence_value,
            "Prior real producer did not complete before the next FG nomination");
          const auto expected_pattern=i ? (initial_pattern^i)&1u : initial_pattern;
          selected->pattern=(initial_pattern^(i+1u))&1u;
          pending_present("SLFG-sustained-latest-pending-prior-complete");
          previous_present_end=GetTickCount64();
          ++pending_frames;
          const auto output=published();
          require(output.generation==previous_publication.generation && output.sequence>previous_publication.sequence &&
            output.timestamp>previous_publication.timestamp,"Sustained FG pipeline stopped publishing current presentations");
          previous_publication=output;
          const bool stereo_ready=captured_ready && ready();
          if(!stereo_ready) {
            ++mono_frames;
            exact_current_mono(output);
          } else {
            ++ready_frames;
            require(captured.source_id==initial_depth.source_id && captured.resource.handle==initial_depth.resource.handle &&
              captured.projection.A==initial_depth.projection.A && captured.projection.B==initial_depth.projection.B &&
              scalar("Sunshine_CameraStrengthBlend")==1.f,
              "Sustained FG changed the logical source, display binding, projection or strength ramp");
            retained_scale_ui(initial_scale,"SLFG-sustained-scale");
            const bool advanced=captured.provided.sequence>previous_real_sequence;
            if(i) {if(advanced)++refreshed_frames;else ++stale_frames;}
            previous_real_sequence=captured.provided.sequence;
            const auto actual_depth=read(reinterpret_cast<ID3D12Resource *>(captured.resource.handle));
            const auto &expected=expected_depth[expected_pattern];
            const bool depth_matches=actual_depth.size()==expected.size()*sizeof(float) &&
              std::memcmp(actual_depth.data(),expected.data(),actual_depth.size())==0;
            if(!depth_matches)++wrong_depth_frames;
            const auto pixels=payload(output);
            float green_error{},eye_difference{};
            for(unsigned y=height/8;y<height*7/8;y+=std::max(1u,height/90)) {
              std::uint16_t expected_half{};
              std::memcpy(&expected_half,source_bytes.data()+size_t(y)*width*8+2,2);
              const float expected=half_float(expected_half);
              for(unsigned x=width/8;x<width*7/8;x+=std::max(1u,width/160)) {
                for(unsigned eye=0;eye<2;++eye)
                  green_error=std::max(green_error,std::abs(channel(pixels,eye*width+x,y,1)-expected));
                eye_difference=std::max(eye_difference,std::abs(channel(pixels,x,y,2)-channel(pixels,width+x,y,2)));
              }
            }
            require(green_error<.003f && eye_difference>.01f,
              "Sustained FG output did not render current color with actual stereo disparity");
            std::printf("MEASURE sustained_FG frame=%u prior_producer_complete=1 latest_pending=1 real_sequence=%llu reused=%u advanced=%u depth_exact=%u current_color_error=%.9g\n",
              i,static_cast<unsigned long long>(captured.provided.sequence),unsigned(captured.reused_depth),
              unsigned(advanced),unsigned(depth_matches),green_error);
          }
          std::printf("MEASURE sustained_FG frame=%u interframe_gap_ms=%llu present_ms=%llu validation_ms=%llu\n",i,
            static_cast<unsigned long long>(interframe_gap),static_cast<unsigned long long>(previous_present_end-frame_started),
            static_cast<unsigned long long>(GetTickCount64()-previous_present_end));
          // GPU presentations and full pixel readbacks already span multiple
          // reuse lifetimes. Do not artificially delay the next 4K capture;
          // the elapsed assertion below verifies the sustained test duration.
        }
        const auto elapsed=GetTickCount64()-started;
        std::printf("MEASURE sustained_FG total=%u pending=%u ready=%u mono=%u prior_real_refreshes=%u stale=%u wrong_depth_frames=%u elapsed_ms=%llu CPU_watchdog_releases=0\n",
          frame_count,pending_frames,ready_frames,mono_frames,refreshed_frames,stale_frames,wrong_depth_frames,
          static_cast<unsigned long long>(elapsed));
        require(elapsed>100 && pending_frames==frame_count,"Sustained FG regression did not span the original 100ms starvation interval");
        require(ready_frames==frame_count && !mono_frames && refreshed_frames==frame_count-1 && !stale_frames && !wrong_depth_frames,
          "A perpetually newer pending FG frame starved completed real depth or discarded it during capture allocation");
        for(unsigned failure_case=0;failure_case<2;++failure_case) {
          // The loop above (and recovery below) leaves real N completed only
          // AFTER effects: it exists in the capture ring but is not displayed.
          // A failed or missing N+1 must invalidate that completed history too.
          const auto completed_before_failure=producer_completion->GetCompletedValue();
          require(completed_before_failure!=UINT64_MAX && completed_before_failure>=producer_fence_value,
            "Failure barrier did not start with a completed unconsumed producer");
          const auto before_failure=published();
          fg_tag_result=failure_case==0?1:0;
          missing_fg_depth=failure_case==1;
          change_color();current_mono_fast(before_failure,failure_case ?
            "SLFG-sustained-missing-depth-barrier" : "SLFG-sustained-failed-tag-barrier");
          fg_tag_result=0;missing_fg_depth=false;
          const auto before_pending=published();
          pending_present("SLFG-sustained-pending-after-failure");
          const auto after_pending=published();
          require(!captured_ready && !ready() && after_pending.sequence>before_pending.sequence,
            "A valid pending frame resurrected a completed depth snapshot from before the FG failure");
          exact_current_mono(after_pending);
          // The pending frame just completed after acquisition. Present without
          // another nomination so it is the first eligible post-failure source.
          emit=false;change_color();ngx_tick("SLFG-sustained-post-failure-completed-recovery");
          require(captured_ready && !captured.reused_depth &&
            read(reinterpret_cast<ID3D12Resource *>(captured.resource.handle))==
              read(boundary_witness.p,D3D12_RESOURCE_STATE_COPY_DEST),
            "A completed post-failure real depth did not recover exact current pixels");
          emit=true;settle_fg("SLFG-sustained-post-failure-camera-recovery");
          if(failure_case==0)pending_present("SLFG-sustained-unconsumed-before-missing-depth");
          std::printf("PASS sustained FG failure barrier case=%s prevents pre-failure completed depth resurrection and admits fresh completion\n",
            failure_case ? "missing depth" : "failed tag");
        }
        selected->pattern=initial_pattern;settle_fg("SLFG-sustained-completed-recovery");
        set_foreground(nullptr);record_fg_tags={};
        std::puts("PASS sustained asynchronous FG publishes current-color stereo using each previous completed real depth while the newest producer stays pending; no depth starvation, calibration restart or age extension");
        return;
      }

      if(sunshine_camera_fixture::flag("SUNSHINE_FG_PENDING_PRODUCER_TEST") || producer_oracle_only) {
        require(cross_queue && complete_producer && retire_producer_recording,
          "FG pending-producer regression requires a warmed, separately queued and retired producer");
        // The ordinary warmup deliberately completes the producer on the CPU.
        // These frames instead remain behind a real GPU fence until the effect
        // observer has seen acquisition. A foreign pending snapshot must stay
        // unavailable without blocking the CPU or adding a GPU dependency.
        // FG may use completed real depth with new color inside its age bound.
        constexpr unsigned frame_count=8;
        unsigned pending_frames{},reused_frames{},expired_frames{};
        size_t total_wrong_witness{};
        const auto initial_pattern=selected->pattern;
        for(unsigned i=0;i<frame_count;++i) {
          // Each case starts with fresh stereo, so inspection of a previous
          // 4K witness cannot consume this case's depth-reuse interval.
          if(!producer_oracle_only)settle_fg("SLFG-before-gated-producer");
          const auto previous_depth_pixels=producer_oracle_only ? std::vector<std::uint8_t>{} :
            read(reinterpret_cast<ID3D12Resource *>(captured.resource.handle));
          if(!producer_oracle_only)ngx_tick("SLFG-refresh-before-gated-producer");
          const auto previous=published();const auto previous_depth=captured;
          const auto before_scale=producer_oracle_only ? scale_display{} : current_scale_ui(1,"SLFG-before-gated-producer");
          const auto before_shader_scale=scalar("Sunshine_CameraDepthScale");
          const auto before_source=captured.source_id;
          // Nominate a different phase while the next GPU capture is pending.
          // This present must keep the phase frozen with the previous depth.
          fg_jitter_x=i&1?.375f:-.125f;fg_jitter_y=i&1?-.25f:.5f;
          if(!producer_oracle_only)require(previous_depth.provided.jitter.x!=fg_jitter_x,
            "Pending producer jitter case did not change the newest nomination");
          const bool expire_depth=!producer_oracle_only && i==frame_count-1;
          if(expire_depth)Sleep(270);
          if(debug_messages.p)debug_messages->ClearStoredMessages();
          com_ptr<ID3D12Fence> gate;
          checked(game->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(gate.put())),"Create deterministic pending FG producer gate");
          HANDLE release_observed=CreateEventW(nullptr,TRUE,FALSE,nullptr);
          require(release_observed,"Create pending FG producer observation event");
          bool observed_pending{},observed_effect{},current{},stereo{};
          HRESULT release_result=E_FAIL;
          std::atomic<bool> emergency_release{false};
          ngx_frame_observer=[&] {
            if(observed_effect)return;
            observed_effect=true;
            const auto completed=producer_completion->GetCompletedValue();
            observed_pending=completed!=UINT64_MAX && completed<producer_fence_value;
            current=captured_ready;
            stereo=ready();
            release_result=gate->Signal(1);SetEvent(release_observed);
          };
          std::thread release_watchdog([&] {
            if(WaitForSingleObject(release_observed,2000)!=WAIT_OBJECT_0) {
              emergency_release=true;gate->Signal(1);
            }
          });
          producer_gated=true;selected->pattern=(initial_pattern^(i+1u))&1u;change_color();
          try {
            checked(producer_queue->Wait(gate.p,1),"Hold FG producer pending until actual effect acquisition");
            ngx_tick("SLFG-current-retired-producer-pending");
          } catch(...) {
            gate->Signal(1);SetEvent(release_observed);release_watchdog.join();
            ngx_frame_observer={};producer_gated=false;CloseHandle(release_observed);
            throw;
          }
          release_watchdog.join();ngx_frame_observer={};producer_gated=false;CloseHandle(release_observed);
          require(!emergency_release && observed_effect && SUCCEEDED(release_result),
            "FG acquisition blocked the CPU waiting for an unfinished producer");
          pending_frames+=observed_pending;
          if(!producer_oracle_only) {
            require(observed_pending,"FG producer did not stay pending until its acquisition observer");
            const auto output=published();
            if(expire_depth) {
              require(!current && !stereo && output.generation==previous.generation && output.sequence>previous.sequence &&
                payload(output)==check_current_mono(),
                "Unfinished FG producer prolonged depth reuse beyond its bounded lifetime");
              ++expired_frames;
            } else {
              require(current && stereo,"Pending FG producer failed to reuse available completed real depth");
              check_fresh_stereo(previous,previous_depth,previous_depth_pixels,"SLFG-gated-producer-reuse");
              retained_scale_ui(before_scale,"SLFG-gated-producer-reuse");
              ++reused_frames;
            }
            require(captured.provided.provider==sunshine_scene_depth::provider_kind::streamline &&
              captured.provided.source_id==before_source && scalar("Sunshine_CameraDepthScale")==before_shader_scale,
              "Unfinished FG producer switched source authority or changed its calculated scale");
          }
          std::printf("MEASURE FG captured native state=0x%x proof=%u selected_post_state=0x%x\n",
            captured.provided.native_state,unsigned(captured.provided.proof),unsigned(selected->state));
          const auto boundary_pixels=read(boundary_witness.p,D3D12_RESOURCE_STATE_COPY_DEST);
          size_t wrong_witness{};
          for(unsigned y=0;y<selected->height;++y)for(unsigned x=0;x<selected->width;++x) {
            float expected=.9375f;
            if(x>=left && x<left+active_width && y>=top && y<top+active_height) {
              const auto cx=std::min(31u,unsigned((double(x-left)+.5)*32/active_width));
              const auto cy=std::min(17u,unsigned((double(y-top)+.5)*18/active_height));
              expected=.0078125f*float((selected->pattern ? 31-cx : cx)/8+1);
              if(cx>=14 && cx<18 && cy>=7 && cy<11)expected=center_raw;
            }
            float actual{};std::memcpy(&actual,boundary_pixels.data()+(size_t(y)*selected->width+x)*4,4);
            wrong_witness+=actual!=expected;
          }
          total_wrong_witness+=wrong_witness;
          std::printf("MEASURE pending producer witness frame=%u pending=%u current=%u stereo=%u wrong_pixels=%zu skip_NGX=%u skip_SL=%u\n",
            i,unsigned(observed_pending),unsigned(current),unsigned(stereo),wrong_witness,
            unsigned(producer_oracle_only || skip_ngx_gated),unsigned(producer_oracle_only));
          if(wrong_witness) {
            const auto path=runtime_directory/("pending-FG-frame-"+std::to_string(i)+".witness.r32f");
            sunshine_parity::write_bytes(path,boundary_pixels.data(),boundary_pixels.size());
          }
          if(producer_oracle_only) {
            const auto poisoned_pixels=read(selected->resource.p,selected->state);
            size_t wrong_poison{};
            for(size_t offset=0;offset<poisoned_pixels.size();offset+=4) {
              float poison{};std::memcpy(&poison,poisoned_pixels.data()+offset,4);wrong_poison+=poison!=.75f;
            }
            std::printf("MEASURE producer-only gate frame=%u pending=%u wrong_witness_pixels=%zu wrong_poison_pixels=%zu native_recording=0x%llx no_NGX_or_SL_capture_calls=1\n",
              i,unsigned(observed_pending),wrong_witness,wrong_poison,static_cast<unsigned long long>(producer_recording_native));
            if(wrong_witness || wrong_poison) {
              sunshine_parity::write_bytes(runtime_directory/"producer-only-witness.r32f",boundary_pixels.data(),boundary_pixels.size());
              sunshine_parity::write_bytes(runtime_directory/"producer-only-poison.r32f",poisoned_pixels.data(),poisoned_pixels.size());
            }
            require(!wrong_witness && !wrong_poison,"Gated producer fixture violated its own full-frame copy-before-poison oracle without SDK capture calls");
            continue;
          }
          // Completion permits a fresh current capture again; the gap must not
          // switch provider or restart the scene-derived scale/strength ramp.
          settle_fg("SLFG-gated-producer-completed-recovery");
          check_jitter({fg_jitter_x,fg_jitter_y,active_width,active_height,true});
          require(captured.source_id==before_source && scalar("Sunshine_CameraDepthScale")==before_shader_scale &&
            scalar("Sunshine_CameraStrengthBlend")==1.f,
            "FG completion recovery changed source/calibration or restarted the strength ramp");
          require(read(reinterpret_cast<ID3D12Resource *>(captured.resource.handle))==
            read(boundary_witness.p,D3D12_RESOURCE_STATE_COPY_DEST),
            "FG completion recovery did not capture the fresh exact SDK-boundary depth");
        }
        std::printf("MEASURE deterministic pending SLFG frames=%u pending=%u previous_depth_reused=%u expired_to_mono=%u wrong_witness_pixels=%zu CPU_watchdog_releases=0\n",
          frame_count,pending_frames,reused_frames,expired_frames,total_wrong_witness);
        require(pending_frames==frame_count,"FG pending-producer gate failed to exercise unfinished GPU work on every frame");
        require(total_wrong_witness==0,"SDK capture changed the independent producer copy-before-poison witness");
        if(producer_oracle_only) {
          set_foreground(nullptr);record_fg_tags={};
          std::puts("PASS producer-only gated GPU execution preserves every witness/source pixel without NGX or SL capture calls; no add-on stereo/readiness verdict made");
          return;
        }
        require(reused_frames==frame_count-1 && expired_frames==1,
          "Pending FG snapshots did not exercise both bounded real-depth reuse and expiry");
        selected->pattern=initial_pattern;
        settle_fg("SLFG-pending-producer-following-completed-frame");
        std::puts("PASS eight unfinished foreign FG producers never expose unfinished pixels or block the CPU; previous real depth renders current-color SBS for at most250ms, then expires to mono; completed captures recover without recalibration");
      }

      if(cross_queue) {
        // Pixel oracles above include large readbacks and artifact writes.
        // Start the bounded-reuse scenario with fresh real depth rather than
        // letting test inspection consume its 250 ms source freshness window.
        settle_fg("FG-fresh-frame-after-pixel-oracles");
        const auto reuse_pending=[&](const char *phase) {
          const auto depth_pixels=read(reinterpret_cast<ID3D12Resource *>(captured.resource.handle));
          // Refresh while the producer is temporarily in its ordinary completed
          // state; the immediately following frame exercises the requested gap.
          const bool submit_requested=submit_producer_recording,retire_requested=retire_producer_recording;
          submit_producer_recording=retire_producer_recording=true;
          ngx_tick("FG-refresh-before-pending-source");
          submit_producer_recording=submit_requested;retire_producer_recording=retire_requested;
          const auto started=GetTickCount64();
          const auto before=published();const auto before_depth=captured;
          const auto before_scale=current_scale_ui(1,"SLFG-before-pending-source");
          const auto previous_frame=fg_frame;const auto submissions=producer_submissions;
          fg_jitter_x=-fg_jitter_x;fg_jitter_y=-fg_jitter_y;
          change_color();ngx_tick(phase);const auto held=published();
          LARGE_INTEGER counter{},frequency{};QueryPerformanceCounter(&counter);QueryPerformanceFrequency(&frequency);
          std::printf("MEASURE %s prior_sequence=%llu current_sequence=%llu inspection_and_present_ms=%llu prior_frame_age_ms=%.3f\n",
            phase,static_cast<unsigned long long>(before.sequence),static_cast<unsigned long long>(held.sequence),
            static_cast<unsigned long long>(GetTickCount64()-started),
            double(counter.QuadPart-before.timestamp)*1000./double(frequency.QuadPart));
          require(fg_frame==previous_frame+1 && (submit_producer_recording || producer_submissions==submissions),
            "Pending FG case did not tag a new frame with the requested native submission state");
          check_fresh_stereo(before,before_depth,depth_pixels,phase);
          retained_scale_ui(before_scale,phase);
        };
        retire_producer_recording=false;
        reuse_pending("FG-cross-queue-replayable-recording");
        retire_producer_recording=true;settle_fg("FG-cross-queue-retired-recovery");
        submit_producer_recording=false;
        reuse_pending("FG-cross-queue-unsubmitted-recording");
        submit_producer_recording=true;settle_fg("FG-cross-queue-submitted-recovery");
        std::puts("PASS valid FG input with replayable or unsubmitted producer renders current-color SBS using exact previous depth and recovers on the next fresh snapshot");

        // Source success and bounded age remain independent of GPU readiness.
        // A newer failed/missing tag must defeat reuse, even if the producer
        // is never submitted; an unfinished valid frame cannot prolong it.
        submit_producer_recording=false;fg_tag_result=1;change_color();
        current_mono(published(),"FG-pending-producer-failed-tag");
        fg_tag_result=0;submit_producer_recording=true;settle_fg("FG-pending-failed-tag-recovery");
        submit_producer_recording=false;missing_fg_depth=true;change_color();
        current_mono(published(),"FG-pending-producer-missing-depth");
        missing_fg_depth=false;submit_producer_recording=true;settle_fg("FG-pending-missing-depth-recovery");
        submit_producer_recording=false;Sleep(270);change_color();
        current_mono(published(),"FG-pending-producer-reuse-expired");
        submit_producer_recording=true;settle_fg("FG-pending-expiry-recovery");
        submit_producer_recording=false;
        reuse_pending("FG-pending-producer-before-disable");
        configure(0,0);emit=false;change_color();current_mono(published(),"FG-disabled-with-unsubmitted-source");
        configure(1,1);emit=true;submit_producer_recording=true;settle_fg("FG-pending-disabled-recovery");
        std::puts("PASS unsubmitted FG input cannot hide failed/missing tags, extend250ms real-depth reuse, or survive FG disable");

        const bool original_call_overlap=sunshine_camera_fixture::flag("SUNSHINE_FG_PENDING_PRODUCER_TEST");
        const bool nomination_overlap=sunshine_camera_fixture::flag("SUNSHINE_FG_PUBLICATION_GAP_TEST");
        if(original_call_overlap || nomination_overlap) {
          using block_tag_t=void (*)(HANDLE,HANDLE);
          const auto block_original=reinterpret_cast<block_tag_t>(GetProcAddress(interposer,"SunshineFixtureBlockNextTag"));
          const auto block_nomination=reinterpret_cast<block_tag_t>(GetProcAddress(GetModuleHandleW(L"SunshineSBSTest.addon64"),"SunshineStreamlineTestNominationGate"));
          require(!original_call_overlap || block_original,"Pending SDK-call regression requires the fixture's original-tag gate");
          require(!nomination_overlap || block_nomination,"Publication-gap regression setup requires a test add-on with the nomination gate; this is not a product failure");
          // A real middleware call can overlap Present before it returns. Use a
          // separate recording and worker so no rendering or command-list state
          // is re-entered. Phase 0 pauses the SDK's original call after capture;
          // phase 1 pauses the real nomination before any snapshot is published.
          // Neither gate injects a source, slot, readiness or exported pixels.
          for(unsigned phase=0;phase<2;++phase) {
            if((phase==0 && !original_call_overlap) || (phase==1 && !nomination_overlap))continue;
            const auto block_tag=phase?block_nomination:block_original;
            const char *gate_name=phase?"nomination-before-slot":"original-SDK-call";
            set_int("Depth_Map_View",phase?2:0);
            for(unsigned scenario=0;scenario<5;++scenario) {
              // Missing tags are rejected before a valid nomination reaches its
              // gate; phase 0 already exercises that independent hard failure.
              if(phase && scenario==2)continue;
              settle_fg("FG-before-overlapping-tag");
              const auto before_pixels=payload(published());
              const auto before_depth_pixels=read(reinterpret_cast<ID3D12Resource *>(captured.resource.handle));
              ngx_tick("FG-refresh-before-overlapping-tag");
              const auto before=published();const auto before_depth=captured;
              const auto before_scale=current_scale_ui(1,"SLFG-before-overlap");
              if(phase)normal_depth_pixels(before_pixels,"before-nomination-gap");
              const bool failure_result=scenario==1,missing_resource=scenario==2;
              com_ptr<ID3D12CommandAllocator> pending_allocator;
              com_ptr<ID3D12GraphicsCommandList> pending_commands;
              checked(game->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(pending_allocator.put())),
                "Create overlapping FG original-call allocator");
              checked(game->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,pending_allocator.p,nullptr,IID_PPV_ARGS(pending_commands.put())),
                "Create overlapping FG original-call recording");
              checked(pending_commands->Close(),"Close initial overlapping FG recording");
              checked(pending_commands->Reset(pending_allocator.p,nullptr),"Observe overlapping FG recording reset");
              const auto pending_native=game_native_command;
              require(pending_native,"Overlapping FG recording did not expose its native command list");
              transition(pending_commands.p,selected->resource.p,selected->state,D3D12_RESOURCE_STATE_COPY_SOURCE);
              transition(pending_commands.p,selected->resource.p,D3D12_RESOURCE_STATE_COPY_SOURCE,selected->state);
              abi_v2::frame_token *token{};const auto index=++fg_frame;
              if(live_compat)token=reinterpret_cast<abi_v2::frame_token *>(opaque_token);
              else require(new_token(token,&index)==0 && token,"Overlapping FG call failed to obtain its frame token");
              abi_v2::constants constants{};constants.base={nullptr,constants_guid,1};
              constants.common.camera_view_to_clip=camera.projection;constants.common.clip_to_camera_view=camera.inverse_projection;
              constants.common.camera_near=camera.near_plane;constants.common.camera_far=camera.far_plane;
              constants.common.camera_fov=camera.fov;constants.common.camera_aspect=camera.aspect;
              constants.common.camera_right[0]=constants.common.camera_up[1]=constants.common.camera_forward[2]=1;
              constants.depth_inverted=1;
              require(set_constants(constants,*token,viewport)==0,"Overlapping FG call changed the constants result");
              abi_v2::resource resource{};resource.base={nullptr,resource_guid,1};resource.type=8;
              resource.native=selected->resource.p;resource.state=unsigned(selected->state);
              resource.width=selected->width;resource.height=selected->height;resource.native_format=DXGI_FORMAT_R32_FLOAT;
              resource.mip_levels=resource.array_layers=1;
              if(live_compat){resource.base={};resource.type=0;resource.state=0;resource.width=resource.height=0;}
              abi_v2::resource_tag tag{{&precision,tag_guid,1},missing_resource?nullptr:&resource,0,0,{top,left,active_width,active_height}};
              HANDLE entered=CreateEventW(nullptr,TRUE,FALSE,nullptr),release=CreateEventW(nullptr,TRUE,FALSE,nullptr);
              if(!entered || !release) {
                if(entered)CloseHandle(entered);if(release)CloseHandle(release);
                require(false,"Create overlapping original-tag synchronization events");
              }
              set_tag_result(failure_result?1:0);block_tag(entered,release);
              std::atomic<bool> returned{false};std::int32_t sdk_result=-1;
              std::thread worker([&] {
                sdk_result=live_compat ? set_global_tag(viewport,&tag,1,reinterpret_cast<void *>(pending_native)) :
                  set_tag(*token,viewport,&tag,1,reinterpret_cast<void *>(pending_native));
                returned.store(true,std::memory_order_release);
              });
              const auto release_worker=[&] {
                SetEvent(release);worker.join();block_tag(nullptr,nullptr);CloseHandle(entered);CloseHandle(release);
              };
              try {
                require(WaitForSingleObject(entered,2000)==WAIT_OBJECT_0 && !returned.load(std::memory_order_acquire),
                  "FG tag did not remain in progress at its requested deterministic overlap gate");
                emit=false;change_color();
                if(missing_resource)current_mono(before,"FG-overlapping-original-missing-depth");
                else {
                  ngx_tick(phase?"FG-overlapping-nomination-before-slot":"FG-overlapping-original-tag-pending");const auto held=published();
                  require(!returned.load(std::memory_order_acquire),"In-progress FG SDK call blocked Present");
                  const auto held_pixels=payload(held);
                  std::printf("MEASURE FG overlap gate=%s scenario=%u prior_sequence=%llu current_sequence=%llu exact_payload=%u sdk_still_pending=%u\n",
                    gate_name,scenario,static_cast<unsigned long long>(before.sequence),static_cast<unsigned long long>(held.sequence),
                    unsigned(held_pixels==before_pixels),unsigned(!returned.load(std::memory_order_acquire)));
                  if(phase)normal_depth_pixels(held_pixels,"inside-nomination-gap");
                  check_fresh_stereo(before,before_depth,before_depth_pixels,"SLFG-inside-overlap",phase!=0);
                  if(phase)require(held_pixels==before_pixels,"Pending FG nomination changed the reused Normal Depth pixels");
                  retained_scale_ui(before_scale,"SLFG-inside-overlap");
                  if(scenario==3) {Sleep(270);change_color();current_mono(held,"FG-overlapping-original-reuse-expired");}
                  if(scenario==4) {configure(0,0);change_color();current_mono(held,"FG-overlapping-original-disabled");}
                }
                require(!returned.load(std::memory_order_acquire),"FG overlap gate watchdog expired during the regression");
              } catch(...) {
                release_worker();emit=true;set_tag_result(0);throw;
              }
              release_worker();
              require(sdk_result==(failure_result?1:0),"Observed overlapping FG tag changed its original SDK result");
              checked(pending_commands->Close(),"Close unsubmitted overlapping FG recording after original return");
              if(failure_result) {change_color();current_mono(published(),"FG-overlapping-original-returned-failure");}
              if(scenario==4)configure(1,1);
              emit=true;set_tag_result(0);
              // Normal Depth does not depend on game color; reverse the actual
              // scene bands to prove recovery consumes a fresh depth snapshot.
              if(phase)selected->pattern^=1;
              settle_fg("FG-overlapping-original-fresh-source-recovery");
              current_scale_ui(1,"SLFG-recovered-from-overlap");
              if(phase) {
                const auto recovered=published();const auto recovered_pixels=payload(recovered);
                normal_depth_pixels(recovered_pixels,"after-nomination-gap-recovery");
                require(recovered.generation==before.generation && recovered.sequence>before.sequence && recovered_pixels!=before_pixels,
                  "Normal Depth recovery did not publish the newly changed scene depth");
              }
              std::printf("PASS FG overlap gate=%s scenario=%u (0=success 1=failure 2=missing 3=expiry 4=off) preserves bounded previous-depth reuse and fresh recovery\n",gate_name,scenario);
            }
          }
          set_int("Depth_Map_View",0);settle_fg("FG-overlap-return-to-game-view");
        }
      }

      for(unsigned count : {1u,3u}) {
        configure(1,count);emit=true;settle_fg("FG-real-source-before-extras");
        const auto depth_pixels=read(reinterpret_cast<ID3D12Resource *>(captured.resource.handle));
        ngx_tick("FG-refresh-before-generated-presents");
        const auto started=GetTickCount64();
        const auto source=published();const auto source_depth=captured;const auto evaluation=sequence;
        const auto before_scale=current_scale_ui(1,"SLFG-before-generated-present");
        emit=false;auto previous=source;
        for(unsigned i=0;i<count;++i) {
          change_color();ngx_tick("FG-extra-no-depth-present");const auto output=published();
          check_reused_depth(source_depth);
          require(output.generation==previous.generation && output.sequence>previous.sequence && output.timestamp>previous.timestamp,
            "An FG extra present did not publish its new-color stereo frame");
          retained_scale_ui(before_scale,"SLFG-generated-present");
          previous=output;
        }
        const auto elapsed=GetTickCount64()-started;
        require(elapsed<250,"FG positive case exceeded its 250ms source freshness budget; rerun this timing-sensitive functional case at 1280x720");
        require(sequence==evaluation,"Generated presentations unexpectedly evaluated a new real depth source");
        check_fresh_stereo(source,source_depth,depth_pixels,"SLFG-generated-present");
        emit=true;ngx_tick("FG-next-real-depth-frame");const auto resumed=published();
        require(resumed.generation==source.generation && resumed.sequence>previous.sequence && sl_current() && ready() && !captured.reused_depth,
          "The next real depth frame did not resume stereo publication immediately");
        require(scalar("Sunshine_CameraStrengthBlend")==1.f,
          "Generated presents restarted stereo entry on the next real frame");
        std::printf("PASS FG %u generated presents publish new-color stereo with exact previous real depth for %llums; next real depth replaces it at full strength\n",
          count,static_cast<unsigned long long>(elapsed));
      }

      // Missing optional jitter is not missing depth or camera data. It must
      // clear the last correction immediately without dropping valid stereo.
      fg_jitter_x=std::numeric_limits<float>::quiet_NaN();
      settle_fg("SLFG-fresh-depth-without-valid-jitter");
      require(sl_current() && ready(),"Invalid optional jitter disabled valid SL depth/projection");
      check_jitter({});
      fg_jitter_x=.375f;fg_jitter_y=-.25f;
      settle_fg("SLFG-fresh-jitter-recovery");
      check_jitter({fg_jitter_x,fg_jitter_y,active_width,active_height,true});
      std::puts("PASS fresh/captured FG jitter remains paired through generated-frame reuse and changed pending nominations; missing jitter clears shader correction without disabling depth");

      missing_fg_depth=true;change_color();current_mono(published(),"FG-real-evaluation-missing-depth");
      missing_fg_depth=false;emit=false;change_color();current_mono(published(),"FG-generated-after-missing-depth");
      emit=true;settle_fg("FG-missing-depth-recovery");
      fg_tag_result=1;change_color();current_mono(published(),"FG-real-failed-tag");
      fg_tag_result=0;emit=false;change_color();current_mono(published(),"FG-generated-after-failed-tag");
      emit=true;settle_fg("FG-failed-tag-recovery");
      configure(0,0);emit=false;change_color();current_mono(published(),"FG-disabled-extra-present");
      configure(1,3);change_color();current_mono(published(),"FG-reenabled-without-new-real-depth");
      emit=true;settle_fg("FG-reenabled-new-real-depth");
      const auto before_timeout=published();emit=false;change_color();Sleep(270);
      current_mono(before_timeout,"FG-expired-source-depth");
      emit=true;settle_fg("FG-timeout-recovery");
      // Focus loss clears real-depth history even if the window returns before
      // the age bound. Foreground is the sole injected observation here; actual
      // game rendering, depth selection and export lifecycle continue normally.
      ngx_tick("FG-real-depth-before-focus-loss");
      emit=false;set_foreground(nullptr);
      ngx_tick("FG-unfocused-generated-present");
      set_foreground(window);change_color();
      ngx_tick("FG-refocused-without-new-real-depth");
      require(!ready() && !captured_ready && payload(published())==check_current_mono(),
        "Restoring focus resurrected prior FG depth without a new completed real capture");
      emit=true;settle_fg("FG-refocus-fresh-real-depth-recovery");
      require(!captured.reused_depth,"Fresh depth after focus recovery remained marked as reused");
      std::puts("PASS a foreground roundtrip invalidates FG depth history until a fresh real capture");
      configure(0,0);ngx_settle("FG-off-returns-NGX-source");
      require(!captured.provided.frame_generation_input && captured.provided.provider==sunshine_scene_depth::provider_kind::ngx,
        "FG off did not return source authority to the current NGX depth");
      check_jitter({});
      current_scale_ui(2,"FG-off-NGX-estimated-scale");
      set_foreground(nullptr);
      record_fg_tags={};
      std::puts("PASS FG real missing/failed depth stays current-color mono; off/on cannot resurrect old real depth; 250ms expiry remains bounded");
    }

    void create_producer_queue() {
      // Called before rendering or after step's main-queue completion. That
      // completion follows the game's producer fence Wait, so all these native
      // objects are idle before retirement, including their last depth copy.
      producer_commands.reset();producer_allocator.reset();producer_retirement_allocator.reset();
      producer_completion.reset();producer_queue.reset();
      producer_fence_value=0;
      D3D12_COMMAND_QUEUE_DESC desc{};desc.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
      checked(game->CreateCommandQueue(&desc,IID_PPV_ARGS(producer_queue.put())),"Create distinct NGX producer queue");
      checked(game->CreateCommandAllocator(desc.Type,IID_PPV_ARGS(producer_allocator.put())),"Create NGX producer allocator");
      if(sunshine_camera_fixture::flag("SUNSHINE_FG_FRESH_RETIRE_ALLOCATOR_TEST"))
        checked(game->CreateCommandAllocator(desc.Type,IID_PPV_ARGS(producer_retirement_allocator.put())),"Create independent empty-recording retirement allocator");
      checked(game->CreateCommandList(0,desc.Type,producer_allocator.p,nullptr,IID_PPV_ARGS(producer_commands.put())),
        "Create NGX producer command list");
      checked(producer_commands->Close(),"Close initial NGX producer command list");
      checked(game->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(producer_completion.put())),"Create game producer fence");
      require(producer_queue.p!=queue.p,"NGX cross-queue fixture reused the presentation queue");
      std::printf("MEASURE NGX producer queue generation=%u separate_from_present=1 independent_retirement_allocator=%u\n",
        ++producer_generation,unsigned(producer_retirement_allocator.p!=nullptr));
    }
    void record_ngx_frame() {
      if(!cross_queue) {
        if(evaluate_without_source_state) {
          const bool saved_emit=emit;emit=false;draw_ngx_frame();emit=saved_emit;
          // Produce real depth first. The following successful SDK evaluation
          // is submitted on a fresh recording with no source state evidence.
          submit(false);begin_commands();evaluate_ngx_frame();
        } else draw_ngx_frame();
        if(record_fg_tags) record_fg_tags();
        return;
      }
      // The base step already reset its presentation list. Preserve that native
      // identity while the actual SDK runs on another real recording. Reuse the
      // depth producer without substituting its public metadata or capture path.
      const auto presentation_command=game_native_command;
      checked(producer_allocator->Reset(),"Reset completed NGX producer allocator");
      checked(producer_commands->Reset(producer_allocator.p,nullptr),"Reset NGX producer command list");
      producer_recording_native=game_native_command;
      require(game_native_command && game_native_command!=presentation_command,
        "NGX producer reset did not expose a distinct native command list");
      std::swap(commands.p,producer_commands.p);
      try {
        draw_ngx_frame();
        if(record_fg_tags) record_fg_tags();
        checked(commands->Close(),"Close actual NGX producer recording");
        if(!submit_producer_recording) {
          // Keep this genuine successful SDK recording closed but unsubmitted.
          // The next frame resets it normally; no GPU completion is fabricated.
          std::swap(commands.p,producer_commands.p);game_native_command=presentation_command;
          return;
        }
        ID3D12CommandList *lists[]{commands.p};
        producer_queue->ExecuteCommandLists(1,lists);
        if(retire_producer_recording) {
          // Reset the LIST, not its still-in-flight allocator. The submitted
          // recording can no longer be replayed to overwrite the add-on's copy
          // after foreign acquisition. The empty new recording is not executed.
          // Diagnostic alternative isolates the empty retirement recording's
          // storage from the submitted producer. It persists across frames and
          // is never Reset while any submission may still refer to it.
          checked(commands->Reset(producer_retirement_allocator.p ? producer_retirement_allocator.p : producer_allocator.p,nullptr),
            "Retire submitted NGX producer recording without resetting its allocator");
          checked(commands->Close(),"Close empty NGX producer recording after retirement");
        }
        checked(producer_queue->Signal(producer_completion.p,++producer_fence_value),"Signal game NGX producer completion");
        if(complete_producer && !producer_gated) {
          // An opt-in deterministic test condition, never an add-on behavior:
          // the real producer has completed before ReShade acquires its copy.
          checked(producer_completion->SetEventOnCompletion(producer_fence_value,completion_event),
            "Observe actual producer completion before cross-queue presentation");
          require(WaitForSingleObject(completion_event,3000)==WAIT_OBJECT_0,"Cross-queue producer did not complete within three seconds");
        }
        // This is the GAME's existing ordering, before current color and Present.
        // It neither injects an add-on wait nor fabricates CPU fence completion.
        // The base step waits for main completion only AFTER actual rendering.
        if(!independent_generated_present)
          checked(queue->Wait(producer_completion.p,producer_fence_value),"Order game presentation after NGX producer");
        ++producer_submissions;
      } catch(...) {
        std::swap(commands.p,producer_commands.p);game_native_command=presentation_command;
        throw;
      }
      std::swap(commands.p,producer_commands.p);game_native_command=presentation_command;
    }
    void create_ngx_compute() {
      const char *source=R"(
cbuffer Settings : register(b0) {
  uint w; uint h; uint pattern; uint reverse;
  uint left; uint top; uint active_w; uint active_h;
  float center_raw;
};
RWTexture2D<float> depth : register(u0);
[numthreads(8,8,1)] void main(uint3 id : SV_DispatchThreadID) {
  if(id.x>=w || id.y>=h) return;
  // The explicit NGX extent is surrounded by a radically different value.
  // Sampling the whole allocation, or copying the wrong origin, must fail.
  float d=.9375;
  if(id.x>=left && id.y>=top && id.x<left+active_w && id.y<top+active_h) {
    float2 uv=(float2(id.xy)-float2(left,top)+.5)/float2(active_w,active_h);
    uint2 cell=min(uint2(31,17),uint2(uv*float2(32,18)));
    bool center=cell.x>=14 && cell.x<18 && cell.y>=7 && cell.y<11;
    d=pattern==2 ? .015625 : .0078125*(float((pattern ? 31-cell.x : cell.x)/8)+1);
    if(center) d=center_raw;
  }
  depth[id.xy]=reverse ? d : 1-d;
})";
      com_ptr<ID3DBlob> shader,errors,signature;
      checked(D3DCompile(source,std::strlen(source),nullptr,nullptr,nullptr,"main","cs_5_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3,0,shader.put(),errors.put()),"Compile NGX padded native depth producer");
      D3D12_DESCRIPTOR_RANGE range{};
      range.RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_UAV; range.NumDescriptors=1;
      D3D12_ROOT_PARAMETER root_parameters[2]{};
      root_parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
      root_parameters[0].Constants.Num32BitValues=9;
      root_parameters[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
      root_parameters[1].DescriptorTable={1,&range};
      D3D12_ROOT_SIGNATURE_DESC desc{};
      desc.NumParameters=2;desc.pParameters=root_parameters;
      checked(D3D12SerializeRootSignature(&desc,D3D_ROOT_SIGNATURE_VERSION_1,signature.put(),errors.put()),
        "Serialize NGX native depth root signature");
      checked(game->CreateRootSignature(0,signature->GetBufferPointer(),signature->GetBufferSize(),
        IID_PPV_ARGS(compute_root.put())),"Create NGX native depth root signature");
      D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline{};
      pipeline.pRootSignature=compute_root.p;
      pipeline.CS={shader->GetBufferPointer(),shader->GetBufferSize()};
      checked(game->CreateComputePipelineState(&pipeline,IID_PPV_ARGS(compute_pipeline.put())),"Create NGX native depth pipeline");
    }
    void create_feature() {
      parameters.width=active_width;parameters.height=active_height;
      parameters.flags=reversed ? 8 : 0;
      require(call_create(reinterpret_cast<void *>(game_native_command),1,&parameters,&feature)==ngx_fixture::success,
        "Synthetic game NGX feature creation failed");
    }
    void create_tracked_pipeline() {
      // Same exact scene and poison padding as the UAV producer, written by
      // actual DSV draws so the shared preservation path sees normal game use.
      const char *source=R"(
cbuffer Settings : register(b0) {
  uint w; uint h; uint pattern; uint reverse;
  uint left; uint top; uint active_w; uint active_h;
  float center_raw;
};
float4 vs(uint id : SV_VertexID) : SV_Position {
  float2 uv=float2((id<<1)&2,id&2);
  return float4(uv*float2(2,-2)+float2(-1,1),0,1);
}
float ps(float4 position : SV_Position) : SV_Depth {
  uint2 id=uint2(position.xy);
  float d=.9375;
  if(id.x>=left && id.y>=top && id.x<left+active_w && id.y<top+active_h) {
    float2 uv=(float2(id)-float2(left,top)+.5)/float2(active_w,active_h);
    uint2 cell=min(uint2(31,17),uint2(uv*float2(32,18)));
    bool center=cell.x>=14 && cell.x<18 && cell.y>=7 && cell.y<11;
    d=pattern==2 ? .015625 : .0078125*(float((pattern ? 31-cell.x : cell.x)/8)+1);
    if(center) d=center_raw;
  }
  return reverse ? d : 1-d;
})";
      com_ptr<ID3DBlob> vertex,pixel,errors,signature;
      checked(D3DCompile(source,std::strlen(source),nullptr,nullptr,nullptr,"vs","vs_5_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3,0,vertex.put(),errors.put()),"Compile API-nominated DSV vertex producer");
      checked(D3DCompile(source,std::strlen(source),nullptr,nullptr,nullptr,"ps","ps_5_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3,0,pixel.put(),errors.put()),"Compile API-nominated DSV depth producer");
      D3D12_ROOT_PARAMETER parameters{};
      parameters.ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
      parameters.Constants.Num32BitValues=9;
      parameters.ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
      D3D12_ROOT_SIGNATURE_DESC signature_desc{};
      signature_desc.NumParameters=1;signature_desc.pParameters=&parameters;
      checked(D3D12SerializeRootSignature(&signature_desc,D3D_ROOT_SIGNATURE_VERSION_1,signature.put(),errors.put()),
        "Serialize API-nominated DSV producer signature");
      checked(game->CreateRootSignature(0,signature->GetBufferPointer(),signature->GetBufferSize(),IID_PPV_ARGS(tracked_root.put())),
        "Create API-nominated DSV producer signature");
      D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline{};
      pipeline.pRootSignature=tracked_root.p;
      pipeline.VS={vertex->GetBufferPointer(),vertex->GetBufferSize()};
      pipeline.PS={pixel->GetBufferPointer(),pixel->GetBufferSize()};
      pipeline.SampleMask=UINT_MAX;
      pipeline.RasterizerState.FillMode=D3D12_FILL_MODE_SOLID;
      pipeline.RasterizerState.CullMode=D3D12_CULL_MODE_NONE;
      pipeline.RasterizerState.DepthClipEnable=TRUE;
      pipeline.DepthStencilState.DepthEnable=TRUE;
      pipeline.DepthStencilState.DepthWriteMask=D3D12_DEPTH_WRITE_MASK_ALL;
      pipeline.DepthStencilState.DepthFunc=D3D12_COMPARISON_FUNC_ALWAYS;
      pipeline.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
      pipeline.SampleDesc.Count=1;pipeline.DSVFormat=DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
      checked(game->CreateGraphicsPipelineState(&pipeline,IID_PPV_ARGS(tracked_pipeline.put())),
        "Create API-nominated packed DSV draw pipeline");
    }
    void draw_tracked_depth(uav_target &source,unsigned draw_count) {
      if(source.state!=D3D12_RESOURCE_STATE_DEPTH_WRITE)
        transition(commands.p,source.resource.p,source.state,D3D12_RESOURCE_STATE_DEPTH_WRITE);
      source.state=D3D12_RESOURCE_STATE_DEPTH_WRITE;
      const auto dsv=source.descriptors->GetCPUDescriptorHandleForHeapStart();
      commands->ClearDepthStencilView(dsv,D3D12_CLEAR_FLAG_DEPTH|D3D12_CLEAR_FLAG_STENCIL,tracked_clear_value(),17,0,nullptr);
      commands->OMSetRenderTargets(0,nullptr,FALSE,&dsv);
      commands->SetGraphicsRootSignature(tracked_root.p);commands->SetPipelineState(tracked_pipeline.p);
      commands->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      const D3D12_VIEWPORT viewport{0,0,float(source.width),float(source.height),0,1};
      const D3D12_RECT scissor{0,0,LONG(source.width),LONG(source.height)};
      commands->RSSetViewports(1,&viewport);commands->RSSetScissorRects(1,&scissor);
      struct {unsigned w,h,pattern,reverse,left,top,active_w,active_h;float center;}
        values{source.width,source.height,source.pattern,unsigned(reversed),left,top,active_width,active_height,center_raw};
      commands->SetGraphicsRoot32BitConstants(0,9,&values,0);
      for(unsigned i=0;i<draw_count;++i) commands->DrawInstanced(3,1,0,0);
      commands->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    }
    void poison_tracked_depth() {
      transition(commands.p,selected->resource.p,selected->state,D3D12_RESOURCE_STATE_DEPTH_WRITE);
      selected->state=D3D12_RESOURCE_STATE_DEPTH_WRITE;
      commands->ClearDepthStencilView(selected->descriptors->GetCPUDescriptorHandleForHeapStart(),
        D3D12_CLEAR_FLAG_DEPTH,tracked_clear_value(),0,0,nullptr);
    }
    void record_state_pressure() {
      if(!state_pressure) return;
      require(pressure_resources.size()==40,"NGX state-pressure resources were not prepared");
      // Complete every split pair before using any resource. These forty real
      // unrelated sources legitimately occupy conservative blocked-state entries
      // until Reset, but cannot invalidate the later ordinary NGX depth source.
      // No unfinished split, fake resource, assumed state or alias is submitted.
      for(const auto &resource : pressure_resources) {
        D3D12_RESOURCE_BARRIER split{};
        split.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        split.Transition={resource->resource.p,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
          D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE};
        split.Flags=D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY;
        commands->ResourceBarrier(1,&split);
        split.Flags=D3D12_RESOURCE_BARRIER_FLAG_END_ONLY;
        commands->ResourceBarrier(1,&split);
        transition(commands.p,resource->resource.p,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      }
      ++pressure_recordings;
    }
    void draw_ngx_frame() {
      if(rotate) selected=(++rotation_index&1u) ? first.get() : second.get();
      draw(*decoy);
      record_state_pressure();
      if(tracked_source) {
        tracked_decoy->pattern=selected->pattern^1u;
        draw_tracked_depth(*tracked_decoy,9);
        draw_tracked_depth(*selected,1);
        transition(commands.p,selected->resource.p,D3D12_RESOURCE_STATE_DEPTH_WRITE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        selected->state=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        evaluate_ngx_frame();
        // The public API reports the original resource while it still contains
        // scene depth. Only a before-clear preservation can retain that scene
        // for the later main-queue ReShade render; the original is now poison.
        poison_tracked_depth();
        return;
      }
      if(selected->state!=D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        transition(commands.p,selected->resource.p,selected->state,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      ID3D12DescriptorHeap *heaps[]{selected->descriptors.p};
      commands->SetDescriptorHeaps(1,heaps);
      commands->SetComputeRootSignature(compute_root.p);
      commands->SetPipelineState(compute_pipeline.p);
      struct { unsigned w,h,pattern,reverse,left,top,active_w,active_h;float center; }
        values{selected->width,selected->height,selected->pattern,unsigned(reversed),left,top,active_width,active_height,center_raw};
      commands->SetComputeRoot32BitConstants(0,9,&values,0);
      commands->SetComputeRootDescriptorTable(1,selected->descriptors->GetGPUDescriptorHandleForHeapStart());
      commands->Dispatch((selected->width+7)/8,(selected->height+7)/8,1);
      // A real observed transition is the source-state evidence. No state value
      // is injected through an NGX parameter or a fixture-only capture entry.
      transition(commands.p,selected->resource.p,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      selected->state=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
      if(use_depth_stencil_crop) {
        auto &source=*depth_stencil_crop;
        if(source.state!=D3D12_RESOURCE_STATE_DEPTH_WRITE)
          transition(commands.p,source.resource.p,source.state,D3D12_RESOURCE_STATE_DEPTH_WRITE);
        const auto dsv=source.descriptors->GetCPUDescriptorHandleForHeapStart();
        commands->ClearDepthStencilView(dsv,D3D12_CLEAR_FLAG_DEPTH|D3D12_CLEAR_FLAG_STENCIL,
          reversed ? .9375f : .0625f,17,0,nullptr);
        const auto edge=[](unsigned cell,unsigned extent,unsigned count) {
          return (cell*extent+count/2-1)/count;
        };
        for(unsigned band=0;band<4;++band) {
          const D3D12_RECT rect{LONG(left+edge(band*8,active_width,32)),LONG(top),
            LONG(left+edge((band+1)*8,active_width,32)),LONG(top+active_height)};
          const float d=.0078125f*float((source.pattern ? 3-band : band)+1);
          commands->ClearDepthStencilView(dsv,D3D12_CLEAR_FLAG_DEPTH,reversed ? d : 1.f-d,0,1,&rect);
        }
        const D3D12_RECT center{LONG(left+edge(14,active_width,32)),LONG(top+edge(7,active_height,18)),
          LONG(left+edge(18,active_width,32)),LONG(top+edge(11,active_height,18))};
        commands->ClearDepthStencilView(dsv,D3D12_CLEAR_FLAG_DEPTH,reversed ? center_raw : 1.f-center_raw,0,1,&center);
        transition(commands.p,source.resource.p,D3D12_RESOURCE_STATE_DEPTH_WRITE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        source.state=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
      }
      evaluate_ngx_frame();
    }
    void evaluate_ngx_frame() {
      if(!armed) return;
      if((producer_oracle_only || skip_ngx_gated) && producer_gated)return;
      require(game_native_command,"NGX fixture lost the real native command-list identity");
      if(!feature && auto_create) create_feature();
      if(!feature || !emit) return;
      parameters.depth=use_depth_stencil_crop ? depth_stencil_crop->resource.p : selected->resource.p;
      parameters.left=left;parameters.top=top;
      parameters.active_width=invalid_extent ? selected->width+1 : active_width;
      parameters.active_height=active_height;
      parameters.evaluation_success=valid_evaluation;
      ++sequence;
      const auto result=call_evaluate(reinterpret_cast<void *>(game_native_command),feature,&parameters,nullptr);
      require(result==(valid_evaluation ? ngx_fixture::success : ngx_fixture::failure),"NGX hook changed the original evaluation result");
    }
    bool ngx_current() const {
      const auto &source=ngx_source();
      return captured_ready && captured_render==observed.renders && !captured.projection.supplied &&
        captured.source_resource.handle==native(source) && captured.width==source.width && captured.height==source.height &&
        captured.active_width==active_width && captured.active_height==active_height && captured.x==left && captured.y==top;
    }
    void ngx_tick(const char *phase) {
      step();
      trace<<phase<<','<<GetTickCount64()<<','<<observed.renders<<','<<sequence<<','<<native(ngx_source())<<','
        <<captured.source_resource.handle<<','<<captured.source_id<<','<<captured.projection.supplied<<','<<captured_ready<<','
        <<ready()<<','<<scalar("Sunshine_CameraDepthScale")<<','<<zero()[1]<<','<<scalar("Sunshine_CameraStrengthBlend")<<'\n';
      require(trace.good(),"Cannot write actual NGX trajectory");
    }
    void ngx_settle(const char *phase,unsigned timeout=15000,bool verify_each_ready_frame=false) {
      const auto started=GetTickCount64();unsigned continuous=0;
      do {
        ngx_tick(phase);
        if(verify_each_ready_frame && ngx_current() && ready()) verify_ngx_depth();
        continuous=ngx_current() && ready() && scalar("Sunshine_CameraStrengthBlend")==1.f ? continuous+1 : 0;
      } while(continuous<6 && GetTickCount64()-started<timeout);
      std::printf("MEASURE %s elapsed_ms=%llu current=%u ready=%u source=%llu H=%.9g t0=%.9g getters=%u\n",phase,
        static_cast<unsigned long long>(GetTickCount64()-started),unsigned(ngx_current()),unsigned(ready()),
        static_cast<unsigned long long>(captured.source_id),scalar("Sunshine_CameraDepthScale"),zero()[1],ngx_fixture::getters);
      require(continuous>=6,"NGX native depth did not reach sustained adaptive rendering");
    }
    void verify_ngx_depth() {
      require(ngx_current() && ready(),"NGX pixel verification requires a current adaptive source");
      int basis=-1;
      observed.runtime->get_uniform_value_int(uniform("Sunshine_CameraCoordinateBasis"),&basis,1);
      require(basis==1 && !captured.projection.supplied,"NGX without camera metadata invented a projection scale");
      require(selected_binding().handle==captured.shader_resource.handle,"NGX current shader binding differs from its real capture");
      const auto &source=ngx_source();
      auto *texture=reinterpret_cast<ID3D12Resource *>(captured.resource.handle);
      require(texture->GetDesc().Width==source.width && texture->GetDesc().Height==source.height,
        "NGX capture changed allocation dimensions instead of carrying its active rectangle");
      const auto bytes=packed_source() ? read_packed_plane(texture,0,D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE) :
        read(texture,D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
      // A legal whole-plane GPU copy retains the poison padding. The source
      // rectangle belongs to sampling, not an illegal partial depth-stencil copy.
      for(unsigned y=0;y<source.height;++y) for(unsigned x=0;x<source.width;++x) {
        float nearness=.9375f;
        if(x>=left && y>=top && x<left+active_width && y<top+active_height) {
          const unsigned cell_x=std::min(31u,unsigned((float(x-left)+.5f)/active_width*32.f));
          const unsigned cell_y=std::min(17u,unsigned((float(y-top)+.5f)/active_height*18.f));
          const bool center=cell_x>=14 && cell_x<18 && cell_y>=7 && cell_y<11;
          const auto band=(source.pattern ? 31-cell_x : cell_x)/8;
          nearness=center ? center_raw : source.pattern==2 ? .015625f : .0078125f*(band+1);
        }
        const float expected=reversed ? nearness : 1.f-nearness;
        float actual{};std::memcpy(&actual,bytes.data()+(size_t(y)*source.width+x)*sizeof(float),sizeof(float));
        if(actual!=expected)
          std::printf("MEASURE NGX depth mismatch sequence=%llu source=%llu pixel=%u,%u expected=%.9g actual=%.9g tracked=%u\n",
            static_cast<unsigned long long>(captured.provided.sequence),static_cast<unsigned long long>(native(source)),
            x,y,expected,actual,unsigned(tracked_source));
        require(actual==expected,"NGX full allocation copy contains stale or incorrect depth or padding");
      }
      if(packed_source()) {
        const auto original=read_packed_plane(source.resource.p,0,source.state);
        if(tracked_source) {
          for(size_t i=0;i<original.size();i+=sizeof(float)) {
            float actual{};std::memcpy(&actual,original.data()+i,sizeof(float));
            require(actual==tracked_clear_value(),"Tracked source did not really clear after API evaluation");
          }
        } else require(original==bytes,"NGX capture modified original packed source depth");
        const auto stencil=read_packed_plane(source.resource.p,1,source.state);
        require(std::all_of(stencil.begin(),stencil.end(),[](auto v){return v==17;}),
          "NGX packed capture modified original source stencil");
      }
      const auto prepared=read(linear_depth.p);const auto desc=linear_depth->GetDesc();
      std::uint16_t half{};std::memcpy(&half,prepared.data()+(size_t(desc.Height/2)*desc.Width+desc.Width/2)*4+2,2);
      require(std::abs(half_float(half)-1.f/(1.f+scalar("Sunshine_CameraDepthScale")*center_raw))<.002f,
        "NGX shader preparation does not use current automatic raw scaling and orientation");
      for(unsigned x : {1u,unsigned(desc.Width)-2}) {
        const unsigned band=source.pattern ? (x<desc.Width/2 ? 3u : 0u) : (x<desc.Width/2 ? 0u : 3u);
        const float nearness=source.pattern==2 ? .015625f : .0078125f*(band+1);
        std::memcpy(&half,prepared.data()+(size_t(desc.Height/2)*desc.Width+x)*4+2,2);
        require(std::abs(half_float(half)-1.f/(1.f+scalar("Sunshine_CameraDepthScale")*nearness))<.002f,
          "Actual shader sampled allocation padding instead of the current NGX active rectangle");
      }
    }
    void require_status(bool selected_expected,bool ready_expected) {
      sunshine_streamline::provider::source_status status;
      require(query_provider_status(observed.runtime,&status) && status.selected==selected_expected && status.ready==ready_expected,
        "NGX UI selected/ready status disagrees with actual accepted capture ownership");
      if(ready_expected)
        require(status.current.resource==native(ngx_source()) && status.current.identity &&
          status.current.identity==sunshine_native_identity::resource_cookie(ngx_source().resource.p) &&
          status.current.width==active_width && status.current.height==active_height,
          "NGX UI reports the wrong current source or extent");
      else require(!status.current.resource && !status.current.identity,"NGX UI marks historical source as active on a missing current frame");
    }
    void run_state_pressure() {
      for(unsigned i=0;i<40;++i) pressure_resources.push_back(target_uav(16,16,0));
      const auto binding=selected_binding();
      std::vector<std::uint8_t> first_output;
      state_pressure=true;
      for(unsigned i=0;i<12;++i) {
        selected->pattern=i&1u; // Current pixels change while center scale stays fixed.
        ngx_tick("NGX-source-state-pressure");
        require(ngx_current() && ready() && captured.source_id==logical_source && selected_binding()==binding &&
          scalar("Sunshine_CameraDepthScale")==64.f && scalar("Sunshine_CameraStrengthBlend")==1.f,
          "Forty unrelated resource states invalidated current NGX depth or its unchanged calibration");
        require_status(true,true);
        if(i==0 || i==11) {
          verify_ngx_depth();
          const auto pixels=read(exported.p);
          if(i==0) first_output=pixels;
          else require(pixels!=first_output,"Native HDR stereo output froze while current NGX depth changed under resource pressure");
        }
      }
      require(pressure_recordings==12,"NGX source-state pressure did not span twelve genuine command-list resets");
      state_pressure=false;
      ngx_settle("NGX-state-pressure-reset-recovery");verify_ngx_depth();
      require(captured.source_id==logical_source && selected_binding()==binding && scalar("Sunshine_CameraDepthScale")==64.f,
        "Ending unrelated state pressure changed the source or retained an invalid recording");
      std::puts("PASS forty unrelated completed split-transition pairs before NGX evaluation preserve exact current 4K HDR depth, shader pixels, binding and calibration across twelve command-list resets");
    }
    void run_cross_queue() {
      const auto binding=selected_binding();
      const unsigned first_submission=producer_submissions;
      std::vector<std::uint8_t> first_output;
      for(unsigned i=0;i<12;++i) {
        selected->pattern=i&1u;
        ngx_tick("NGX-separate-producer-current-frame");
        require(ngx_current() && ready() && captured.source_id==logical_source && selected_binding()==binding &&
          scalar("Sunshine_CameraDepthScale")==64.f && scalar("Sunshine_CameraStrengthBlend")==1.f,
          "Game-ordered separate NGX queue lost current depth, stable binding or calibration");
        require_status(true,true);
        if(i==0 || i==11) {
          verify_ngx_depth();
          const auto pixels=read(exported.p);
          if(i==0) first_output=pixels;
          else require(pixels!=first_output,"Separate-queue NGX HDR output froze while producer depth changed");
        }
      }
      require(producer_submissions==first_submission+12,
        "Separate NGX producer did not execute twelve genuine recordings and game queue waits");
      retire_producer_recording=false;
      for(unsigned i=0;i<3;++i) {
        ngx_tick("NGX-completed-producer-still-replayable");
        require(!captured_ready && !ready(),"A completed but replayable NGX producer exposed mutable depth to another queue");
        require_status(true,false);
      }
      check_current_mono();
      retire_producer_recording=true;
      ngx_settle("NGX-retired-producer-recording-recovery");verify_ngx_depth();
      require(captured.source_id==logical_source && selected_binding()==binding && scalar("Sunshine_CameraDepthScale")==64.f,
        "Retiring the replayable NGX recording changed source, calibration or binding");
      std::puts("PASS completed but replayable producer recording remains mono; retiring it allows a fresh immutable capture without recalibration");
      check_pending_cross_queue();
      emit=false;
      for(unsigned i=0;i<4;++i) {
        ngx_tick("NGX-separate-producer-missing-evaluation");
        require(!captured_ready && !ready(),"Separate producer queue exposed historical depth after a missing NGX evaluation");
        require_status(true,false);
      }
      check_current_mono();
      emit=true;ngx_settle("NGX-separate-producer-gap-recovery");verify_ngx_depth();
      // Repeated real queue lifetimes catch retained queue/fence ownership. The
      // main presentation queue stays alive; replacing it needs a new swapchain
      // and is a different lifecycle from the producer that owns NGX work.
      for(unsigned i=0;i<3;++i) {
        create_producer_queue();
        selected->pattern=i&1u;
        ngx_settle("NGX-producer-queue-recreation");verify_ngx_depth();require_status(true,true);
        require(captured.source_id==logical_source && selected_binding()==binding && scalar("Sunshine_CameraDepthScale")==64.f,
          "NGX producer queue recreation changed logical source, calibration or stable shader binding");
      }
      require(producer_generation==4,"NGX producer queue retirement was not exercised across four native queue lifetimes");
      std::puts("PASS game-ordered separate NGX producer queue drives exact current 4K HDR depth and shader pixels across resets, missing-frame mono/recovery and four producer queue lifetimes");
    }
    void check_pending_continuity() {
      namespace wire = reshade_bridge;
      require(cross_queue && complete_producer && retire_producer_recording,
        "NGX continuity requires a genuine separate retired producer queue");
      const auto set_foreground=reinterpret_cast<void (*)(HWND)>(GetProcAddress(
        GetModuleHandleW(L"SunshineSBSTest.addon64"),"SunshineSbsTestSetForeground"));
      require(set_foreground,"NGX continuity requires the existing test foreground observer");
      struct foreground_scope {void (*set)(HWND);~foreground_scope(){set(nullptr);}} foreground{set_foreground};
      set_foreground(window);
      struct mapping_guard {
        HANDLE handle{};wire::shared_state_t *state{};
        ~mapping_guard(){if(state)UnmapViewOfFile(state);if(handle)CloseHandle(handle);}
      } mapping;
      const auto mapping_name=std::wstring(wire::mapping_prefix)+std::to_wstring(GetCurrentProcessId());
      mapping.handle=OpenFileMappingW(FILE_MAP_ALL_ACCESS,FALSE,mapping_name.c_str());
      require(mapping.handle,"NGX continuity cannot open actual exporter mapping");
      mapping.state=static_cast<wire::shared_state_t *>(MapViewOfFile(mapping.handle,FILE_MAP_ALL_ACCESS,0,0,sizeof(wire::shared_state_t)));
      require(mapping.state,"NGX continuity cannot map actual exporter state");
      InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&mapping.state->consumer_nonce),0x4e475850454e4449);
      const auto read64=[](std::uint64_t &value){return std::uint64_t(InterlockedCompareExchange64(reinterpret_cast<volatile LONG64 *>(&value),0,0));};
      std::uint64_t last_publication{},generation{};
      const auto output=[&] {
        const auto before=InterlockedCompareExchange(reinterpret_cast<volatile LONG *>(&mapping.state->metadata_sequence),0,0);
        require(!(before&1),"NGX exporter metadata was being changed after completed Present");
        const auto metadata=mapping.state->metadata;MemoryBarrier();
        require(before==InterlockedCompareExchange(reinterpret_cast<volatile LONG *>(&mapping.state->metadata_sequence),0,0) &&
          wire::valid_metadata(metadata) && metadata.dxgi_format==10 && metadata.color_transfer==wire::transfer::scrgb &&
          metadata.source_width==width && metadata.source_height==height,"NGX continuity export lost coherent HDR metadata");
        unsigned index=wire::slot_count;std::uint64_t sequence{};
        for(unsigned i=0;i<wire::slot_count;++i) {
          const auto control=read64(mapping.state->slots[i].control),candidate=read64(mapping.state->slots[i].sequence);
          if(wire::control_generation(control)==metadata.generation && wire::control_state(control)==wire::slot_state::ready && candidate>sequence)
            {index=i;sequence=candidate;}
        }
        require(index<wire::slot_count && sequence>last_publication && (!generation || generation==metadata.generation),
          "NGX continuity retained an old export instead of publishing current color");
        auto &slot=mapping.state->slots[index];
        const auto ready_control=wire::slot_control(metadata.generation,wire::slot_state::ready);
        const auto reading_control=wire::slot_control(metadata.generation,wire::slot_state::reading);
        require(std::uint64_t(InterlockedCompareExchange64(reinterpret_cast<volatile LONG64 *>(&slot.control),reading_control,ready_control))==ready_control &&
          read64(slot.sequence)==sequence,"NGX continuity could not claim its actual exported pixels");
        const auto release=[&]{InterlockedCompareExchange64(reinterpret_cast<volatile LONG64 *>(&slot.control),ready_control,reading_control);};
        try {
          com_ptr<ID3D12Fence> fence;com_ptr<ID3D12Resource> texture;
          checked(game->OpenSharedHandle(reinterpret_cast<HANDLE>(metadata.ready_fence_handle),IID_PPV_ARGS(fence.put())),"Open NGX exported fence");
          require(fence->GetCompletedValue()!=UINT64_MAX && fence->GetCompletedValue()>=sequence,"NGX export preceded GPU completion");
          checked(game->OpenSharedHandle(reinterpret_cast<HANDLE>(metadata.texture_handles[index]),IID_PPV_ARGS(texture.put())),"Open NGX exported texture");
          auto pixels=read(texture.p,D3D12_RESOURCE_STATE_COMMON);release();
          require(pixels==read(exported.p),"NGX exporter did not publish the actual current shader result");
          generation=metadata.generation;last_publication=sequence;return pixels;
        } catch(...) {release();throw;}
      };
      com_ptr<ID3D12Resource> alternate_upload;
      auto alternate_bytes=source_bytes;
      for(size_t i=0;i<alternate_bytes.size();i+=8) {
        const std::uint16_t red=0x4200,green=0x3400;
        std::memcpy(alternate_bytes.data()+i,&red,2);std::memcpy(alternate_bytes.data()+i+2,&green,2);
      }
      D3D12_PLACED_SUBRESOURCE_FOOTPRINT alternate_footprint{};
      fill_upload(alternate_upload,backbuffers[0]->GetDesc(),alternate_bytes.data(),alternate_footprint);
      require(alternate_footprint.Footprint.RowPitch==source_footprint.Footprint.RowPitch,"Alternate NGX source color changed upload layout");
      const auto change_color=[&]{std::swap(source_upload.p,alternate_upload.p);source_bytes.swap(alternate_bytes);};
      for(bool reset : {false,true}) {
        parameters.reset=0;
        ngx_settle("NGX-continuity-fresh-seed");verify_ngx_depth();
        const auto original_depth=read(reinterpret_cast<ID3D12Resource *>(captured.resource.handle));
        // Full-resolution readback may exceed the age bound. Refresh the exact
        // same scene once before starting the measured pending presentation.
        ngx_tick("NGX-continuity-refresh-after-readback");
        const auto previous=captured;
        const auto scale=scalar("Sunshine_CameraDepthScale");const auto convergence=zero();
        require(previous.ready && ready() && !previous.reused_depth,"NGX continuity seed is not a fresh stereo capture");
        com_ptr<ID3D12Fence> gate;
        checked(game->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(gate.put())),"Create NGX continuity producer gate");
        HANDLE done=CreateEventW(nullptr,TRUE,FALSE,nullptr);require(done,"Create NGX continuity cleanup event");
        std::atomic<bool> rescued{false};
        std::thread watchdog([&]{if(WaitForSingleObject(done,5000)!=WAIT_OBJECT_0){rescued=true;gate->Signal(1);}});
        const auto cleanup=[&] {
          gate->Signal(1);SetEvent(done);watchdog.join();CloseHandle(done);
          ngx_frame_observer={};producer_gated=false;independent_generated_present=false;parameters.reset=0;
          render_tracked_depth=[&]{record_ngx_frame();};
          checked(producer_completion->SetEventOnCompletion(producer_fence_value,completion_event),"Observe continuity producer cleanup");
          require(WaitForSingleObject(completion_event,3000)==WAIT_OBJECT_0,"Continuity producer did not finish after fixture gate release");
        };
        try {
          bool observed{},pending{};
          ngx_frame_observer=[&] {
            if(observed)return;
            observed=true;const auto complete=producer_completion->GetCompletedValue();
            pending=complete!=UINT64_MAX && complete<producer_fence_value;
          };
          independent_generated_present=true;producer_gated=true;parameters.reset=reset?1:0;
          selected->pattern^=1u;change_color();
          checked(producer_queue->Wait(gate.p,1),"Keep NGX producer pending across two independent presentations");
          ngx_tick(reset?"NGX-pending-reset-mono":"NGX-pending-one-presentation-hold");
          require(!rescued && observed && pending,"NGX acquisition waited for pending work or failed to exercise a real pending producer");
          const auto first_output=output();
          if(reset) {
            require(!captured_ready && !ready() && !captured.reused_depth && first_output==check_current_mono(),
              "A pending NGX reset reused old depth or old stereo");
          } else {
            require(captured_ready && ready() && captured.reused_depth && captured.frame_index==previous.frame_index+1 &&
              captured.resource==previous.resource && captured.provided.sequence==previous.provided.sequence &&
              captured.provided.tick==previous.provided.tick && captured.provided.feedback.revision==previous.provided.feedback.revision &&
              scalar("Sunshine_CameraDepthScale")==scale && zero()==convergence && scalar("Sunshine_CameraStrengthBlend")==1.f,
              "Known pending NGX copy did not retain exactly one completed depth/geometry pair");
            require(read(reinterpret_cast<ID3D12Resource *>(captured.resource.handle))==original_depth,
              "NGX pending hold sampled newer unfinished depth instead of the owned completed display");
            float green_error{},stereo_difference{};
            for(unsigned y=height/8;y<height*7/8;y+=std::max(1u,height/90))
              for(unsigned x=width/8;x<width*7/8;x+=std::max(1u,width/160)) {
                std::uint16_t expected{};std::memcpy(&expected,source_bytes.data()+(size_t(y)*width+x)*8+2,2);
                for(unsigned eye=0;eye<2;++eye)
                  green_error=std::max(green_error,std::abs(channel(first_output,eye*width+x,y,1)-half_float(expected)));
                stereo_difference=std::max(stereo_difference,std::abs(channel(first_output,x,y,2)-channel(first_output,width+x,y,2)));
              }
            require(green_error<.003f && stereo_difference>.01f,"NGX hold exported old color or mono instead of current-color stereo");
            std::printf("MEASURE NGX one-presentation hold current_color_error=%.9g stereo_difference=%.9g sequence=%llu depth_sequence=%llu\n",
              green_error,stereo_difference,static_cast<unsigned long long>(last_publication),static_cast<unsigned long long>(captured.provided.sequence));
            // Do not reset the gated producer allocator or issue another depth
            // call. The same successful pending input spans a second Present.
            require(GetTickCount64()>=previous.provided.tick &&
              GetTickCount64()-previous.provided.tick<sunshine_scene_depth::maximum_source_age_ms,
              "Fixture readback exhausted source freshness before the second-presentation bound could be tested");
            render_tracked_depth=[]{};change_color();
            ngx_tick("NGX-pending-second-presentation-mono");
            require(!captured_ready && !ready() && !captured.reused_depth && output()==check_current_mono(),
              "NGX held old depth beyond one subsequent native presentation");
          }
          require(!rescued && producer_completion->GetCompletedValue()<producer_fence_value,
            "NGX display/export completion depended on releasing the pending producer");
        } catch(...) {cleanup();throw;}
        cleanup();
        ngx_settle("NGX-continuity-fresh-recovery");verify_ngx_depth();
        require(captured_ready && ready() && !captured.reused_depth && captured.provided.sequence>previous.provided.sequence,
          "Fresh completed NGX input did not recover after bounded hold/reset");
        require(output()==read(exported.p),"Recovered NGX did not export fresh stereo");
        std::printf("PASS NGX continuity reset=%u: pending GPU remained gated through completed current-color exports; fresh depth recovered; watchdog_releases=0\n",unsigned(reset));
      }
    }

    void check_pending_cross_queue() {
      const auto previous_scale=scalar("Sunshine_CameraDepthScale");
      const auto previous_binding=selected_binding();
      selected->pattern^=1u;
      com_ptr<ID3D12Fence> gate;
      checked(game->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(gate.put())),"Create pending producer admission gate");
      HANDLE release_observed=CreateEventW(nullptr,TRUE,FALSE,nullptr);
      require(release_observed,"Create pending producer release event");
      bool observed_pending=false,producer_unfinished=false,published_depth=false,published_camera=false;
      HRESULT release_result=E_FAIL;
      std::atomic<bool> emergency_release{false};
      // Release the GAME's gate only after effect setup has declined the
      // unfinished foreign copy. A CPU completion wait requires the watchdog
      // and fails the test; no add-on GPU dependency should be necessary.
      ngx_frame_observer=[&] {
        if(observed_pending) return;
        observed_pending=true;
        const auto completed=producer_completion->GetCompletedValue();
        producer_unfinished=completed!=UINT64_MAX && completed<producer_fence_value;
        published_depth=ngx_current();published_camera=ready();
        release_result=gate->Signal(1);
        SetEvent(release_observed);
      };
      std::thread release_watchdog([&] {
        if(WaitForSingleObject(release_observed,2000)!=WAIT_OBJECT_0) {
          emergency_release=true;gate->Signal(1);
        }
      });
      producer_gated=true;
      try {
        checked(producer_queue->Wait(gate.p,1),"Hold actual NGX producer pending until the effect observes it");
        ngx_tick("NGX-separate-producer-pending-gate");
      } catch(...) {
        gate->Signal(1);SetEvent(release_observed);release_watchdog.join();
        ngx_frame_observer={};producer_gated=false;CloseHandle(release_observed);
        throw;
      }
      release_watchdog.join();ngx_frame_observer={};producer_gated=false;CloseHandle(release_observed);
      require(!emergency_release && observed_pending && SUCCEEDED(release_result),
        "Pending NGX producer blocked the CPU before effect setup");
      require(producer_unfinished && !published_depth && !published_camera && !captured_ready,
        "Unfinished foreign NGX producer exposed depth pixels or rendered stereo");
      require_status(true,false);check_current_mono();
      require(captured.provided.provider==sunshine_scene_depth::provider_kind::ngx &&
        captured.provided.source_id==logical_source && scalar("Sunshine_CameraDepthScale")==previous_scale &&
        selected_binding()==previous_binding,
        "Pending NGX producer lost source authority, stable binding or estimated calibration");
      ngx_settle("NGX-separate-producer-pending-recovery");verify_ngx_depth();
      require(captured.source_id==logical_source && scalar("Sunshine_CameraDepthScale")==previous_scale &&
        selected_binding()==previous_binding && scalar("Sunshine_CameraStrengthBlend")==1.f,
        "Completed NGX recovery changed logical source, binding, calibration or full-strength stereo");
      std::puts("PASS unfinished foreign NGX producer stays current-color mono without a CPU wait, preserves source/calibration, and recovers exact depth and stereo after completion");
    }
    void run_cross_queue_async() {
      // Before the first accepted NGX copy, Generic legitimately still owns
      // depth. Establish ownership using real completed/retired producer work,
      // then remove only the fixture's CPU completion wait for this phase.
      complete_producer=true;
      ngx_settle("NGX-async-established-owner-warmup");verify_ngx_depth();require_status(true,true);
      logical_source=captured.source_id;
      const auto binding=selected_binding();
      require(logical_source && scalar("Sunshine_CameraDepthScale")==64.f,
        "Asynchronous NGX regression did not first establish its own adaptive source");
      complete_producer=false;
      unsigned captures=0,camera_frames=0,mono_frames=0;
      for(unsigned i=0;i<32;++i) {
        selected->pattern=i&1u;
        ngx_tick("NGX-separate-producer-async-admission");
        if(captured_ready) {
          require(ngx_current() && captured.source_id==logical_source && selected_binding()==binding,
            "Asynchronous NGX producer exposed a previous source, Generic fallback or invalid current extent");
          ++captures;require_status(true,true);
        } else require_status(true,false);
        require(scalar("Sunshine_CameraDepthScale")==64.f,"Asynchronous NGX admission changed the established depth calibration");
        if(ready()) {
          require(ngx_current(),"Asynchronous NGX camera rendered without a current admitted copy");
          ++camera_frames;verify_ngx_depth();
        } else {
          ++mono_frames;
          if(i==0 || i==31) check_current_mono();
        }
      }
      std::printf("PASS asynchronous game-ordered NGX queue admitted only current sources: captures=%u stereo=%u mono=%u; readiness is conservatively optional until the producer completes\n",
        captures,camera_frames,mono_frames);
    }
    void require_tracked_source() {
      require(ngx_current() && ready() && captured.source_id==logical_source &&
        captured.provided.provider==sunshine_scene_depth::provider_kind::ngx &&
        captured.provided.source_id==logical_source && captured.provided.resource.native==native(*selected) &&
        scalar("Sunshine_CameraDepthScale")==64.f && scalar("Sunshine_CameraStrengthBlend")==1.f,
        "Exact API-nominated tracked source lost its current scene, NGX identity or calibration");
      require_status(true,true);
    }
    void run_tracked_source() {
      require(cross_queue && !complete_producer && !retire_producer_recording && tracked_source,
        "Tracked-source regression must use the game's asynchronous queue ordering without early recording retirement");
      unsigned preservation_mode{};
      const auto expected_mode=sunshine_camera_fixture::flag("SUNSHINE_DEPTH_BIND_SWITCH_TEST") ? 2u : 1u;
      require(reshade::get_config_value(nullptr,"DEPTH","DepthCopyBeforeClears",preservation_mode) && preservation_mode==expected_mode,
        "Tracked-source regression did not load the requested preservation mode");
      std::printf("MEASURE NGX tracked-source DepthCopyBeforeClears=%u\n",preservation_mode);
      ngx_settle("NGX-tracked-source-preservation-warmup");verify_ngx_depth();require_status(true,true);
      const auto shared_demand=capture_demand(true);
      logical_source=captured.source_id;
      require(logical_source,"Tracked API source has no stable logical NGX identity");
      const auto binding=selected_binding();
      std::vector<std::uint8_t> first_output;
      for(unsigned i=0;i<32;++i) {
        selected->pattern=i&1u;
        ngx_tick("NGX-tracked-source-current-32");require_tracked_source();
        require(capture_demand(true)==shared_demand,"Steady shared preservation changed Generic capture demand");
        require(selected_binding()==binding,"Tracked API source recreated its shader binding during normal frames");
        verify_ngx_depth();
        if(i==0 || i==31) {
          const auto pixels=read(exported.p);
          if(i==0) first_output=pixels;
          else require(pixels!=first_output,"Tracked API source produced frozen HDR stereo while current scene depth changed");
        }
      }
      std::puts("PASS API-selected packed depth gives 32/32 current stereo frames through game-ordered shared preservation, despite a matching-size higher-draw-count decoy and poisoned original");

      selected=second.get();ngx_settle("NGX-tracked-second-member-warmup");verify_ngx_depth();require_tracked_source();
      selected=first.get();ngx_settle("NGX-tracked-first-member-return");verify_ngx_depth();require_tracked_source();
      first->pattern=0;second->pattern=1;rotate=true;
      ngx_settle("NGX-tracked-rotation-warmup");unsigned seen=0;
      for(unsigned i=0;i<16;++i) {
        ngx_tick("NGX-tracked-rotation-current");require_tracked_source();verify_ngx_depth();
        require(selected_binding()==binding,"Tracked resource rotation changed the stable shader binding");
        seen|=selected==first.get()?1u:2u;
      }
      require(seen==3,"Tracked API-source rotation did not render both physical members");
      rotate=false;selected=first.get();
      std::puts("PASS rotating API-selected tracked resources retain current exact pixels, logical source and adaptive scale");

      const auto require_missing=[&](const char *phase) {
        for(unsigned i=0;i<4;++i) {
          ngx_tick(phase);
          require(!captured_ready && !ready(),"Missing or failed tracked API frame exposed stale scene depth or a matching decoy");
          require_status(true,false);
          require(scalar("Sunshine_CameraDepthScale")==64.f,"Missing tracked API frame changed the established depth scale");
        }
        check_current_mono();
      };
      valid_evaluation=false;require_missing("NGX-tracked-failed-evaluation");
      valid_evaluation=true;ngx_settle("NGX-tracked-failure-recovery");verify_ngx_depth();require_tracked_source();
      parameters.provide_depth=false;require_missing("NGX-tracked-missing-depth");
      parameters.provide_depth=true;ngx_settle("NGX-tracked-depth-recovery");verify_ngx_depth();require_tracked_source();
      emit=false;require_missing("NGX-tracked-no-evaluation");
      emit=true;ngx_settle("NGX-tracked-evaluation-recovery");verify_ngx_depth();require_tracked_source();
      std::puts("PASS failed, missing-depth and absent API frames hold NGX ownership and current-color mono while valid tracked decoys continue drawing");

      // A non-DSV resource has no preservation inventory. Keep the same feature
      // and raw encoding, and exercise the independent native-copy fallback
      // under its completed + retired cross-queue admission requirements.
      // Bracket this transport handover after the deliberate missing/evaluation
      // failures above; their metadata-only nominations can change demand too.
      const auto before_native_demand=capture_demand(true);
      auto saved_first=std::move(first),saved_second=std::move(second);
      first=target_uav(active_width+7,active_height+5,0);
      second=target_uav(active_width+7,active_height+5,1);selected=first.get();
      tracked_source=false;complete_producer=true;retire_producer_recording=true;
      ngx_settle("NGX-tracked-to-standalone-UAV");verify_ngx_depth();require_tracked_source();
      const auto native_demand=capture_demand(false);
      if(sunshine_camera_fixture::flag("SUNSHINE_CAPTURE_DEMAND_TEST")) {
        std::printf("MEASURE capture demand shared_initial=%llu shared_after_recovery=%llu native=%llu\n",
          static_cast<unsigned long long>(shared_demand),static_cast<unsigned long long>(before_native_demand),
          static_cast<unsigned long long>(native_demand));
        require(native_demand==before_native_demand+1,"Shared-to-native handover did not disable capture exactly once");
      }
      require(!manual(observed.runtime,native(*selected)),"Standalone API UAV unexpectedly entered the tracked DSV inventory");
      for(unsigned i=0;i<4;++i) {
        selected->pattern=i&1u;
        ngx_tick("NGX-standalone-UAV-current");verify_ngx_depth();require_tracked_source();
        require(capture_demand(false)==native_demand,"Native API frames toggled Generic capture back on");
      }
      first=std::move(saved_first);second=std::move(saved_second);selected=first.get();
      tracked_source=true;complete_producer=false;retire_producer_recording=false;
      ngx_settle("NGX-standalone-to-tracked-return",15000,true);verify_ngx_depth();require_tracked_source();
      if(sunshine_camera_fixture::flag("SUNSHINE_CAPTURE_DEMAND_TEST"))
        require(capture_demand(true)==native_demand+1,"Native-to-shared handover did not bootstrap preservation exactly once");
      std::puts("PASS one NGX logical source changes shared-preservation/native-UAV transport and returns without recalibrating or changing source authority");
      same_source_provider_parity();
    }
    void same_source_provider_parity() {
      const auto describe=[&](const char *phase) {
        float projection[2]{},rect[4]{};int basis=-1;
        observed.runtime->get_uniform_value_float(uniform("Sunshine_CameraProjection"),projection,2);
        observed.runtime->get_uniform_value_float(uniform("Sunshine_CameraDepthRect"),rect,4);
        observed.runtime->get_uniform_value_int(uniform("Sunshine_CameraCoordinateBasis"),&basis,1);
        std::printf("MEASURE parity %s source=%llu allocation=%ux%u crop=%u,%u,%u,%u orientation=%u detected=%u basis=%d A=%.9g inverseB=%.9g H=%.9g reference=%.9g t0=%.9g blend=%.9g rect=%.9g,%.9g,%.9g,%.9g prepared=%llux%u\n",
          phase,static_cast<unsigned long long>(captured.source_resource.handle),captured.width,captured.height,
          captured.x,captured.y,captured.active_width,captured.active_height,unsigned(captured.orientation),unsigned(captured.detected_orientation),
          basis,projection[0],projection[1],scalar("Sunshine_CameraDepthScale"),zero()[0],zero()[1],
          scalar("Sunshine_CameraStrengthBlend"),rect[0],rect[1],rect[2],rect[3],
          static_cast<unsigned long long>(linear_depth->GetDesc().Width),linear_depth->GetDesc().Height);
      };
      const auto compare_prepared=[&](const std::vector<std::uint8_t> &api,const std::vector<std::uint8_t> &generic,bool report=true) {
        require(api.size()==generic.size(),"Provider parity changed prepared-depth allocation size");
        std::array<size_t,2> count{};std::array<float,2> maximum{};unsigned printed=0;size_t metadata_differences=0;
        const auto desc=linear_depth->GetDesc();
        for(size_t i=0;i<api.size();i+=2) {
          std::uint16_t a{},b{};std::memcpy(&a,api.data()+i,2);std::memcpy(&b,generic.data()+i,2);
          if(a==b) continue;
          const auto channel=(i/2)%2;const auto pixel=i/4,x=pixel%desc.Width,y=pixel/desc.Width;
          // Mod_Z explicitly repurposes G in each corner (uv within two
          // backbuffer pixels on both axes) as temporal/weapon metadata. It is
          // not scene depth. Keep every R pixel and every actual G depth pixel.
          const bool horizontal_corner=(2*x+1)*width<4*desc.Width ||
            (2*(desc.Width-x)-1)*width<4*desc.Width;
          const bool vertical_corner=(2*y+1)*height<4*desc.Height ||
            (2*(desc.Height-y)-1)*height<4*desc.Height;
          if(channel==1 && horizontal_corner && vertical_corner) {++metadata_differences;continue;}
          ++count[channel];
          maximum[channel]=std::max(maximum[channel],std::abs(half_float(a)-half_float(b)));
          if(report && printed++<8) std::printf("MEASURE parity mismatch pixel=%llu,%llu channel=%zu API=%.9g Generic=%.9g half=%04x/%04x\n",
            static_cast<unsigned long long>(pixel%desc.Width),static_cast<unsigned long long>(pixel/desc.Width),channel,
            half_float(a),half_float(b),unsigned(a),unsigned(b));
        }
        if(report) std::printf("MEASURE parity prepared scene mismatch count=%zu,%zu maximum=%.9g,%.9g excluded_corner_metadata=%zu HDR_checked_next=1\n",
          count[0],count[1],maximum[0],maximum[1],metadata_differences);
        if(report && (count[0] || count[1])) {
          sunshine_parity::write_bytes(runtime_directory/"parity-api-prepared.rg16f",api.data(),api.size());
          sunshine_parity::write_bytes(runtime_directory/"parity-generic-prepared.rg16f",generic.data(),generic.size());
        }
        return !count[0] && !count[1];
      };
      // The Generic provider observes the actual raster viewport. Give both
      // providers that same full-allocation crop and an ordinary far clear,
      // rather than comparing different padding or depth-direction contracts.
      require(call_release(feature)==ngx_fixture::success,"Parity setup could not release the cropped NGX feature");
      feature=nullptr;
      parity_scene=true;left=top=0;active_width=selected->width;active_height=selected->height;
      selected->pattern=0;
      // The replacement feature is genuinely created with this new input size;
      // an evaluation never exceeds the dimensions declared at CreateFeature.
      ngx_settle("NGX-same-source-parity-control");verify_ngx_depth();
      logical_source=captured.source_id;require_tracked_source();
      describe("API");
      const auto api_prepared=read(linear_depth.p),api_stereo=read(exported.p);
      require(call_release(feature)==ngx_fixture::success,"Parity test could not explicitly release NGX authority");
      feature=nullptr;auto_create=false;emit=false;
      require(manual(observed.runtime,native(*selected)),"Parity test could not pin the exact API-selected DSV for Generic");
      const auto settle_generic=[&](const char *phase) {
        const auto started=GetTickCount64();unsigned continuous=0;
        do {
          ngx_tick(phase);require_status(false,false);
          continuous=ngx_current() && ready() && scalar("Sunshine_CameraDepthScale")==64.f &&
            zero()[1]==.015625f && scalar("Sunshine_CameraStrengthBlend")==1.f ? continuous+1 : 0;
        } while(continuous<6 && GetTickCount64()-started<15000);
        require(continuous>=6,"Generic did not establish the same exact-source numeric basis for provider parity");
      };
      settle_generic("Generic-same-source-parity-warmup");
      const auto manual_demand=capture_demand(true);
      // Read the real shader outputs, whose resource states are independent of
      // whether the common owner internally borrowed or copied its raw backup.
      const auto generic_prepared=read(linear_depth.p),generic_stereo=read(exported.p);
      describe("Generic");
      require(compare_prepared(api_prepared,generic_prepared),"Provider choice changed actual prepared depth for the same source and matched calibration");
      require(generic_stereo==api_stereo,"Provider choice changed actual HDR stereo for the same source and matched calibration");
      selected->pattern=1;settle_generic("Generic-same-source-changing-pixels");
      require(capture_demand(true)==manual_demand,"An unchanged manual depth pin repeatedly restarted capture");
      require(!compare_prepared(generic_prepared,read(linear_depth.p),false) && read(exported.p)!=generic_stereo,
        "Generic parity reused stale depth or HDR pixels when the exact source changed");
      selected->pattern=0;settle_generic("Generic-same-source-restored-pixels");
      require(compare_prepared(api_prepared,read(linear_depth.p)) && read(exported.p)==api_stereo,
        "Generic source did not restore the same actual depth/HDR output after a real pixel change");
      std::puts("PASS API release to exact manual Generic source preserves identical prepared depth and HDR stereo at matched calibration; changing source pixels proves freshness");
    }
    void run_ngx() {
      // Model a genuine SDK boundary. GCC otherwise uses same-TU knowledge of
      // the tiny fixture stubs' clobbers across a detourable call (including R10),
      // even with noinline. An installed native hook follows the full Win64 ABI.
      const auto executable=GetModuleHandleW(nullptr);
      call_create=reinterpret_cast<decltype(&NVSDK_NGX_D3D12_CreateFeature)>(GetProcAddress(executable,"NVSDK_NGX_D3D12_CreateFeature"));
      call_evaluate=reinterpret_cast<decltype(&NVSDK_NGX_D3D12_EvaluateFeature)>(GetProcAddress(executable,"NVSDK_NGX_D3D12_EvaluateFeature"));
      call_release=reinterpret_cast<decltype(&NVSDK_NGX_D3D12_ReleaseFeature)>(GetProcAddress(executable,"NVSDK_NGX_D3D12_ReleaseFeature"));
      require(call_create && call_evaluate && call_release,"NGX fixture cannot resolve its actual SDK exports");
      tracked_source_test=tracked_source=sunshine_camera_fixture::flag("SUNSHINE_NGX_TRACKED_SOURCE_TEST");
      create_pipeline();create_ngx_compute();
      if(tracked_source_test) create_tracked_pipeline();
      active_width=width==3840 ? 2227 : width/2-1;active_height=height==2160 ? 1253 : height/2-1;
      left=3;top=2;
      first=tracked_source_test ? target_packed(active_width+7,active_height+5,0) : target_uav(active_width+7,active_height+5,0);
      second=tracked_source_test ? target_packed(active_width+7,active_height+5,1) : target_uav(active_width+7,active_height+5,1);selected=first.get();
      if(tracked_source_test) tracked_decoy=target_packed(active_width+7,active_height+5,1);
      depth_stencil_crop=target_packed(active_width+7,active_height+5,0);
      decoy=target(width,height,1,9,false,true);
      valid_evaluation=false;
      cross_queue=tracked_source_test || sunshine_camera_fixture::flag("SUNSHINE_NGX_CROSS_QUEUE_TEST");
      complete_producer=sunshine_camera_fixture::flag("SUNSHINE_NGX_CROSS_QUEUE_COMPLETED_TEST");
      require(!complete_producer || cross_queue,"Completed NGX producer option requires the cross-queue test");
      require(!tracked_source_test || !complete_producer,"Tracked-source regression must not force producer CPU completion");
      if(tracked_source_test) retire_producer_recording=false;
      require(!cross_queue || !sunshine_camera_fixture::flag("SUNSHINE_NGX_STATE_PRESSURE_TEST"),
        "Run distinct NGX queue and state-pressure regressions separately");
      if(cross_queue) create_producer_queue();
      trace.open(runtime_directory/"ngx-depth-trajectory.csv");
      trace<<std::setprecision(17)<<"phase,wall_ms,render,evaluation,tagged,current,logical_source,projection,capture_ready,camera_ready,H,t0,blend\n";
      reshade::register_event<reshade::addon_event::reset_command_list>(observe_reset);
      render_tracked_depth=[&]{record_ngx_frame();};
      // The feature is created on the first game recording, before the first
      // effect has rendered. Startup discovery must catch Create itself; a
      // later evaluation cannot reconstruct the feature's depth convention.
      armed=true;
      const auto timeout=GetTickCount64()+45000;
      while((!observed.runtime || !observed.renders) && GetTickCount64()<timeout) step();
      require(observed.runtime && observed.renders && !observed.inject,"NGX HDR runtime failed to initialize");
      check_unified_addon();
      const auto module=GetModuleHandleW(L"SunshineSBSTest.addon64");
      manual=reinterpret_cast<select_t>(GetProcAddress(module,"SunshineDepthTestSelectManual"));
      recenter=reinterpret_cast<action_t>(GetProcAddress(module,"SunshineGame3DTestRecalibrate"));
      query_frame=reinterpret_cast<frame_t>(GetProcAddress(module,"SunshineDepthTestFrame"));
      query_provider_status=reinterpret_cast<provider_status_t>(GetProcAddress(module,"SunshineDepthTestProviderStatus"));
      require(manual && recenter && query_frame && query_provider_status,"NGX fixture requires passive frame/UI observation adapters");
      if(sunshine_camera_fixture::flag("SUNSHINE_NGX_PENDING_CONTINUITY_TEST")) {
        // Exercise the actual FX reference and exporter callback path in this
        // existing fixture. Native-only rendering has its separate fixture.
        const auto set_enabled=reinterpret_cast<BOOL (*)(api::effect_runtime *,BOOL)>(GetProcAddress(module,"SunshineGame3DTestSetEnabled"));
        require(set_enabled && set_enabled(observed.runtime,FALSE),"Pending NGX fixture could not select its explicit FX reference path");
      }
      reshade::register_event<reshade::addon_event::reshade_render_technique>(observe_ngx_source);
      set_int("Depth_Map_View",0);set_float("Depth_Adjustment",100);set_float("Sharpen_Power",0);
      find_texture("DoubleTex",exported,width*2,DXGI_FORMAT_R16G16B16A16_FLOAT);
      find_texture("texzBufferN_P",linear_depth,0,DXGI_FORMAT_R16G16_FLOAT);

      // Failed evaluations may be seen, but must not establish provider ownership.
      const auto invalid_until=GetTickCount64()+1500;
      do {ngx_tick("initial-failed-evaluation");require_status(false,false);} while(GetTickCount64()<invalid_until);
      require(ngx_fixture::created==1 && ngx_fixture::getters>0,"Production NGX module discovery did not intercept real exported entry points");
      std::puts("PASS first failed NGX evaluations do not take ownership from Generic");

      // Exercise the real callback accounting between GPU submissions, before
      // an API owns selection. The hook restores its scratch state and demand.
      using accounting_t=BOOL (*)(api::effect_runtime *,std::uint64_t);
      const auto accounting=reinterpret_cast<accounting_t>(GetProcAddress(module,"SunshineDepthTestActivityAccounting"));
      require(accounting && accounting(observed.runtime,native(*depth_stencil_crop)),
        "Hidden candidate accounting, UI expiry or preservation recovery failed");
      std::puts("PASS candidate counters pause while hidden, visible UI resumes them, expiry stops them, and preservation activity remains valid");

      valid_evaluation=true;
      invalid_extent=true;
      for(unsigned i=0;i<4;++i) {ngx_tick("initial-invalid-extent");require_status(false,false);}
      invalid_extent=false;
      std::puts("PASS successful NGX calls with an invalid depth extent do not take ownership from Generic");
      if(!cross_queue) {
        evaluate_without_source_state=true;ngx_tick("first-valid-NGX-without-native-state");evaluate_without_source_state=false;
        require_status(true,false);
        require(!captured_ready && !ready(),"A valid API source without current capture proof rendered stale or Generic depth");
        check_current_mono();
        std::puts("PASS successful valid NGX resource establishes source authority even without native copy state; current output remains mono until capture is ready");
      }
      if(tracked_source_test) {
        run_tracked_source();
        render_tracked_depth={};
        reshade::unregister_event<reshade::addon_event::reshade_render_technique>(observe_ngx_source);
        reshade::unregister_event<reshade::addon_event::reset_command_list>(observe_reset);
        return;
      }
      if(cross_queue && !complete_producer) {
        run_cross_queue_async();
        render_tracked_depth={};
        reshade::unregister_event<reshade::addon_event::reshade_render_technique>(observe_ngx_source);
        reshade::unregister_event<reshade::addon_event::reset_command_list>(observe_reset);
        return;
      }
      if(sunshine_camera_fixture::flag("SUNSHINE_NGX_PENDING_CONTINUITY_TEST")) {
        check_pending_continuity();
        render_tracked_depth={};
        reshade::unregister_event<reshade::addon_event::reshade_render_technique>(observe_ngx_source);
        reshade::unregister_event<reshade::addon_event::reset_command_list>(observe_reset);
        return;
      }
      ngx_settle(cross_queue ? "NGX-completed-separate-producer" : "ngx-low-padded-UAV");verify_ngx_depth();require_status(true,true);
      const auto native_demand=capture_demand(false);
      for(unsigned i=0;i<4;++i) {
        ngx_tick("NGX-stable-native-demand");require_status(true,true);
        require(capture_demand(false)==native_demand,"Steady native API capture toggled Generic work or repeated cleanup");
      }
      if(sunshine_camera_fixture::flag("SUNSHINE_CAPTURE_DEMAND_TEST"))
        std::puts("PASS native API frames keep Generic capture dormant without enable/disable transitions");
      require(!manual(observed.runtime,native(*selected)),"NGX-only non-DSV source unexpectedly entered Generic inventory");
      require(std::abs(scalar("Sunshine_CameraDepthScale")-64.f)<1e-5f && zero()[1]==.015625f,
        "NGX depth without projection did not initialize the shared adaptive center scale");
      logical_source=captured.source_id;
      require(logical_source,"NGX logical depth source identity is missing");
      std::puts("PASS real lower-resolution NGX UAV with poisoned padded extent drives correct HDR adaptive stereo without Generic inventory");

      if(sunshine_camera_fixture::flag("SUNSHINE_NGX_FRAME_GENERATION_TEST")) {
        require(!tracked_source_test,"Run FG scheduling separately from Generic preservation cases");
        run_frame_generation();
        render_tracked_depth={};
        reshade::unregister_event<reshade::addon_event::reshade_render_technique>(observe_ngx_source);
        reshade::unregister_event<reshade::addon_event::reset_command_list>(observe_reset);
        return;
      }

      if(cross_queue || sunshine_camera_fixture::flag("SUNSHINE_NGX_STATE_PRESSURE_TEST")) {
        if(cross_queue) run_cross_queue();else run_state_pressure();
        render_tracked_depth={};
        reshade::unregister_event<reshade::addon_event::reshade_render_technique>(observe_ngx_source);
        reshade::unregister_event<reshade::addon_event::reset_command_list>(observe_reset);
        return;
      }

      const auto stable_binding=selected_binding();
      // A first tag identifies this resource only after its earlier transition.
      // Until a later observed transition, its native state is unproven. Hold
      // the known encoding and show mono instead of guessing that state.
      selected=second.get();
      ngx_tick("new-NGX-resource-first-sighting");
      require(!captured_ready && !ready() && scalar("Sunshine_CameraDepthScale")==64.f,
        "First unproven NGX resource did not hold calibration and remain mono");
      require_status(true,false);check_current_mono();
      ngx_settle("new-NGX-resource-state-recovery");verify_ngx_depth();
      require(captured.source_id==logical_source && scalar("Sunshine_CameraDepthScale")==64.f,
        "First-sighting recovery recalibrated the unchanged NGX encoding");
      std::puts("PASS first resource sighting remains mono until an observed state transition; recovery retains logical calibration");
      rotate=true;ngx_settle("NGX-rotation-warmup");unsigned seen=0;
      for(unsigned i=0;i<16;++i) {
        ngx_tick("rotating-NGX-resources");
        require(ngx_current() && ready() && captured.source_id==logical_source && selected_binding()==stable_binding &&
          scalar("Sunshine_CameraDepthScale")==64.f,"NGX rotation reset calibration, binding or logical feature identity");
        require_status(true,true);seen|=selected==first.get()?1u:2u;
        if(i<2) verify_ngx_depth();
      }
      require(seen==3,"NGX rotation did not capture both physical resources");
      rotate=false;selected=first.get();
      auto replacement=target_uav(first->width,first->height,0);
      first=std::move(replacement);selected=first.get();
      ngx_settle("resource-recreation");verify_ngx_depth();
      require(captured.source_id==logical_source && scalar("Sunshine_CameraDepthScale")==64.f,
        "NGX physical resource recreation changed the stable logical depth domain");
      std::puts("PASS NGX physical rotation and recreation retain logical source, calibration and stable shader binding");

      use_depth_stencil_crop=true;
      ngx_settle("packed-NGX-padded-depth");verify_ngx_depth();require_status(true,true);
      require(captured.source_id==logical_source && scalar("Sunshine_CameraDepthScale")==64.f,
        "Packed NGX depth allocation changed the logical raw encoding or scale");
      std::puts("PASS padded packed D32S8 depth uses legal whole-plane capture, exact sampling rectangle and unchanged original depth/stencil");
      use_depth_stencil_crop=false;
      ngx_settle("packed-to-UAV-NGX-return");verify_ngx_depth();

      valid_evaluation=false;
      for(unsigned i=0;i<4;++i) {ngx_tick("failed-established-evaluation");require(!captured_ready && !ready(),"Failed NGX frame reused stale depth");require_status(true,false);}
      check_current_mono();
      valid_evaluation=true;ngx_settle("failed-evaluation-recovery");
      parameters.provide_depth=false;
      for(unsigned i=0;i<4;++i) {ngx_tick("missing-established-depth");require(!captured_ready && !ready(),"NGX without depth fell back to an unrelated Generic scene");require_status(true,false);}
      check_current_mono();
      parameters.provide_depth=true;ngx_settle("missing-depth-recovery");
      emit=false;
      const auto silent_until=GetTickCount64()+1800;
      do {ngx_tick("silent-established-provider");require(!captured_ready && !ready(),"Silent NGX provider exposed stale depth or Generic fallback");require_status(true,false);}
      while(GetTickCount64()<silent_until);
      check_current_mono();
      emit=true;ngx_settle("silent-provider-recovery");
      require(captured.source_id==logical_source,"NGX temporary gap changed its logical depth domain");
      std::puts("PASS established NGX failures and missing depth hold source ownership and produce current-color mono, then recover");

      // A center change starts smooth refinement with its fresh measurements.
      // The same smoothed screen plane determines the stereo reference.
      center_raw=.03125f;
      const float initial_gain=scalar("Sunshine_CameraDepthScale");
      const auto adapt_start=GetTickCount64(),adapt_until=adapt_start+5000;
      do {
        ngx_tick("zero-plane-NGX-reference");const float gain=scalar("Sunshine_CameraDepthScale");
        require(ngx_current() && ready() && captured.source_id==logical_source && gain<=initial_gain && gain>=32.f &&
          std::abs(gain*zero()[1]-1.f)<2e-6f,
          "NGX broke zero-plane normalization, changed logical source or lost readiness");
        require_status(true,true);
      } while(GetTickCount64()<adapt_until);
      require(std::abs(zero()[1]-.03125f)<1e-5f && std::abs(scalar("Sunshine_CameraDepthScale")-32.f)<.04f,
        "NGX screen-plane reference did not forget the initial scene");
      std::printf("MEASURE NGX screen-plane reference initial=%.9g final=%.9g zero=%.9g\n",
        initial_gain,scalar("Sunshine_CameraDepthScale"),zero()[1]);
      std::puts("PASS NGX reference follows the same smoothed zero without changing source readiness");

      selected->pattern=2;
      require(recenter(observed.runtime),"NGX explicit recalibrate action was rejected");
      ngx_settle("constant-NGX-background");verify_ngx_depth();
      require(std::abs(scalar("Sunshine_CameraDepthScale")-32.f)<1e-5f && zero()[1]==.03125f,
        "NGX recalibration did not initialize fresh shared raw reference");
      set_float("Depth_Adjustment",0);for(unsigned i=0;i<4;++i)ngx_tick("strength-zero");
      const auto mono_pixels=check_current_mono();
      set_float("Depth_Adjustment",50);ngx_settle("strength-half");
      const auto half_shift=shifts(mono_pixels,read(exported.p),"NGX-50");
      set_float("Depth_Adjustment",100);ngx_settle("strength-full");
      const auto full_pixels=read(exported.p);
      const auto full_shift=shifts(mono_pixels,full_pixels,"NGX-100");
      for(unsigned eye=0;eye<2;++eye) {
        require(std::abs(half_shift[eye])>=.5f && std::abs(full_shift[eye])>std::abs(half_shift[eye])+.5f &&
          std::abs(full_shift[eye]-2.f*half_shift[eye])<=1.f,"NGX actual HDR stereo does not respond proportionally to linear user strength");
      }
      sunshine_parity::write_bytes(runtime_directory/"ngx-final.sbs",full_pixels.data(),full_pixels.size());
      std::puts("PASS actual NGX HDR output preserves zero-strength color and restores proportional 0/50/100 strength");

      require(call_release(feature)==ngx_fixture::success,"NGX feature release failed");
      feature=nullptr;auto_create=false;
      for(unsigned i=0;i<4;++i) {
        ngx_tick("explicit-NGX-release");require_status(false,false);
        require(!captured_ready || captured.source_resource.handle!=native(*selected),
          "Released NGX feature still supplied historical depth as current");
      }
      std::puts("PASS explicit NGX feature release returns source ownership to Generic; temporary missing frames remain a distinct held state");
      auto_create=true;reversed=false;selected->pattern=0;center_raw=.015625f;
      ngx_settle("normal-depth-new-feature");verify_ngx_depth();
      require(captured.source_id!=logical_source && scalar("Sunshine_CameraDepthScale")==64.f,
        "New NGX feature with a different depth convention inherited the old source calibration");
      require(ngx_fixture::created==2 && ngx_fixture::released==1 && ngx_fixture::evaluated==sequence,
        "NGX production interception changed native call counts");
      std::puts("PASS NGX feature recreation carries a new logical domain and supports normal depth without camera projection");
      render_tracked_depth={};
      reshade::unregister_event<reshade::addon_event::reshade_render_technique>(observe_ngx_source);
      reshade::unregister_event<reshade::addon_event::reset_command_list>(observe_reset);
      std::puts("PASS production NGX SDK discovery, native copy/submission, adaptive scale, current source UI and real HDR shader regression");
    }
  };
}

int main(int argc,char **argv) {
  std::setvbuf(stdout,nullptr,_IONBF,0);
  if(argc!=5 && argc!=7) {std::fputs("usage: ngx_depth_runtime <official.dll> <Shaders> <test.addon64> <fresh-output> [width height]\n",stderr);return 2;}
  std::thread([]{Sleep(300000);std::fputs("FAIL NGX runtime watchdog\n",stderr);TerminateProcess(GetCurrentProcess(),124);}).detach();
  try {
    width=argc==7?unsigned(std::stoul(argv[5])):3840;height=argc==7?unsigned(std::stoul(argv[6])):2160;
    require(width>=640 && width<=3840 && height>=360 && height<=2160 && width%4==0 && height%4==0,"Invalid NGX fixture dimensions");
    require(sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC") && sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC_ACTIONS_TEST") &&
      (!sunshine_camera_fixture::flag("SUNSHINE_DEPTH_BIND_SWITCH_TEST") || sunshine_camera_fixture::flag("SUNSHINE_NGX_TRACKED_SOURCE_TEST")),
      "NGX fixture requires Automatic and TEST ONLY actions; mode2 is available for the tracked-source variant");
    require(!fs::exists(fs::absolute(argv[4])),"Fresh isolated NGX runtime output is required");
    if(sunshine_camera_fixture::flag("SUNSHINE_D3D12_DEBUG_TEST")) {
      // Opt-in diagnostic only, before ReShade or any D3D12 device is created.
      // Missing Windows Graphics Tools is a setup failure, never installed here.
      const auto d3d12=LoadLibraryExW(L"d3d12.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
      require(d3d12,"D3D12 debug diagnostic cannot load the existing system runtime");
      const auto get_debug=reinterpret_cast<decltype(&D3D12GetDebugInterface)>(GetProcAddress(d3d12,"D3D12GetDebugInterface"));
      com_ptr<ID3D12Debug> debug;
      require(get_debug && SUCCEEDED(get_debug(IID_PPV_ARGS(debug.put()))) && debug.p,
        "D3D12 debug layer was requested but is unavailable; no installation attempted");
      debug->EnableDebugLayer();
      std::puts("SETUP D3D12 debug layer enabled before device creation (functional diagnostics only)");
    }
    if(sunshine_camera_fixture::flag("SUNSHINE_NGX_FRAME_GENERATION_TEST")) {
      const auto stub=fs::absolute(argv[0]).parent_path()/"frame_generation_interposer.dll";
      const auto destination=fs::absolute(argv[4])/"sl.interposer.dll";
      require(fs::is_regular_file(stub),"Build the frame-generation SDK fixture DLL beside this executable");
      fs::create_directories(destination.parent_path());fs::copy_file(stub,destination);
      const auto sdk=LoadLibraryW(destination.c_str());
      require(sdk,"Cannot load isolated frame-generation metadata fixture");
      if(sunshine_camera_fixture::flag("SUNSHINE_FG_LIVE_COMPAT_TEST")) {
        const auto getter=reinterpret_cast<sunshine_streamline::abi_v2::get_feature_function>(GetProcAddress(sdk,"slGetFeatureFunction"));
        require(getter && getter(1000,"slDLSSGSetOptions",cached_fg_options)==0 && cached_fg_options,
          "Cannot cache actual SDK FG options before loading ReShade and add-on");
        std::puts("SETUP cached actual FG options pointer before loading ReShade/add-on");
      }
    }
    ngx_runtime_fixture fixture;fixture.runtime_directory=fs::absolute(argv[4]);
    fixture.initialize(fs::absolute(argv[1]),fs::absolute(argv[2]),fixture.runtime_directory,2,0,fs::absolute(argv[3]));
    if(sunshine_camera_fixture::flag("SUNSHINE_D3D12_DEBUG_TEST"))
      checked(fixture.game->QueryInterface(IID_PPV_ARGS(fixture.debug_messages.put())),"Requested D3D12 debug layer did not expose its diagnostic queue");
    fixture.run_ngx();return 0;
  } catch(const std::exception &error) {std::fprintf(stderr,"FAIL %s\n",error.what());return 1;}
}
