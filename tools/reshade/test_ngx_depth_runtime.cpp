// SPDX-License-Identifier: GPL-3.0-only
// Actual exported NGX calls -> production adapter -> native D3D12 copy/fences ->
// shared adaptive controller -> native Game 3D with no installed FX. The
// synthetic SDK reports metadata only; no capture, depth binding, readiness or
// pixels are injected.
#define SUNSHINE_DIRECT_RUNTIME_FIXTURE_ONLY
#include "test_streamline_direct_runtime.cpp"
#include "test_game3d_native_observation.h"
#include <atomic>
#include <d3d12sdklayers.h>

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
  // Runs inside the game's Present, after native Game 3D acquired this frame's
  // depth and before the fixture waits for its own queue.
  std::function<void()> ngx_frame_observer;
  void observe_ngx_present(api::command_queue *,api::swapchain *) {if(ngx_frame_observer) ngx_frame_observer();}

  struct ngx_runtime_fixture : direct_fixture {
    com_ptr<ID3D12InfoQueue> debug_messages;
    native_game3d_observer game3d{*this};
    ngx_fixture::parameters parameters;
    void *feature{};
    unsigned active_width{},active_height{},left{},top{};
    bool armed{},reversed=true,invalid_extent{},use_depth_stencil_crop{},auto_create=true,evaluate_without_source_state{};
    std::unique_ptr<uav_target> depth_stencil_crop;
    std::vector<std::unique_ptr<uav_target>> pressure_resources;
    bool state_pressure{};
    unsigned pressure_recordings{};
    bool cross_queue{},complete_producer{},producer_gated{},retire_producer_recording=true;
    com_ptr<ID3D12CommandQueue> producer_queue;
    com_ptr<ID3D12CommandAllocator> producer_allocator;
    com_ptr<ID3D12GraphicsCommandList> producer_commands;
    com_ptr<ID3D12Fence> producer_completion;
    std::uint64_t producer_fence_value{};
    unsigned producer_submissions{},producer_generation{};
    std::uint64_t logical_source{};
    // The established logical source's adaptive scale, held through gaps.
    float reference_scale{};
    // The latest present's passive observations.
    automatic_status status;
    sunshine_streamline::provider::source_status provider;
    unsigned presents{},unready_presents{};
    // The latest fresh provider copy: when its present returned, and its sequence.
    std::uint64_t present_started{},fresh_ended{},fresh_sequence{};
    decltype(&NVSDK_NGX_D3D12_CreateFeature) volatile call_create{};
    decltype(&NVSDK_NGX_D3D12_EvaluateFeature) volatile call_evaluate{};
    decltype(&NVSDK_NGX_D3D12_ReleaseFeature) volatile call_release{};
    const uav_target &ngx_source() const { return use_depth_stencil_crop ? *depth_stencil_crop : *selected; }

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

    void create_producer_queue() {
      // Called before rendering or after step's main-queue completion. That
      // completion follows the game's producer fence Wait, so all these native
      // objects are idle before retirement, including their last depth copy.
      producer_commands.reset();producer_allocator.reset();
      producer_completion.reset();producer_queue.reset();
      producer_fence_value=0;
      D3D12_COMMAND_QUEUE_DESC desc{};desc.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
      checked(game->CreateCommandQueue(&desc,IID_PPV_ARGS(producer_queue.put())),"Create distinct NGX producer queue");
      checked(game->CreateCommandAllocator(desc.Type,IID_PPV_ARGS(producer_allocator.put())),"Create NGX producer allocator");
      checked(game->CreateCommandList(0,desc.Type,producer_allocator.p,nullptr,IID_PPV_ARGS(producer_commands.put())),
        "Create NGX producer command list");
      checked(producer_commands->Close(),"Close initial NGX producer command list");
      checked(game->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(producer_completion.put())),"Create game producer fence");
      require(producer_queue.p!=queue.p,"NGX cross-queue fixture reused the presentation queue");
      std::printf("MEASURE NGX producer queue generation=%u separate_from_present=1\n",++producer_generation);
    }
    void record_ngx_frame() {
      if(!cross_queue) {
        if(evaluate_without_source_state) {
          const bool saved_emit=emit;emit=false;draw_ngx_frame();emit=saved_emit;
          // Produce real depth first. The following successful SDK evaluation
          // is submitted on a fresh recording with no source state evidence.
          submit(false);begin_commands();evaluate_ngx_frame();
        } else draw_ngx_frame();
        return;
      }
      // The base step already reset its presentation list. Preserve that native
      // identity while the actual SDK runs on another real recording. Reuse the
      // depth producer without substituting its public metadata or capture path.
      const auto presentation_command=game_native_command;
      checked(producer_allocator->Reset(),"Reset completed NGX producer allocator");
      checked(producer_commands->Reset(producer_allocator.p,nullptr),"Reset NGX producer command list");
      require(game_native_command && game_native_command!=presentation_command,
        "NGX producer reset did not expose a distinct native command list");
      std::swap(commands.p,producer_commands.p);
      try {
        draw_ngx_frame();
        checked(commands->Close(),"Close actual NGX producer recording");
        ID3D12CommandList *lists[]{commands.p};
        producer_queue->ExecuteCommandLists(1,lists);
        if(retire_producer_recording) {
          // Reset the LIST, not its still-in-flight allocator. The submitted
          // recording can no longer be replayed to overwrite the add-on's copy
          // after foreign acquisition. The empty new recording is not executed.
          checked(commands->Reset(producer_allocator.p,nullptr),"Retire submitted NGX producer recording without resetting its allocator");
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

    // Per present: a fresh provider copy of the expected NGX source and extent.
    bool present_current() const {
      return provider.selected && provider.ready && !provider.reused_depth && provider.provider==sunshine_scene_depth::provider_kind::ngx &&
        provider.current.resource==native(ngx_source()) && provider.current.width==active_width && provider.current.height==active_height;
    }
    // The game resource consumed by one native render; rotation allows either member.
    const uav_target *consumed_source(const native_render &frame) const {
      if(use_depth_stencil_crop) return frame.source_resource==native(*depth_stencil_crop) ? depth_stencil_crop.get() : nullptr;
      if(frame.source_resource==native(*selected)) return selected;
      if(rotate) for(const uav_target *member : {first.get(),second.get()}) if(frame.source_resource==native(*member)) return member;
      return nullptr;
    }
    bool ngx_current(const native_render &frame) const {
      const auto *source=consumed_source(frame);
      return source && frame.depth_ready && !frame.reused_depth && frame.provider=="ngx" && !frame.projection_supplied &&
        frame.width==source->width && frame.height==source->height && frame.active_width==active_width &&
        frame.active_height==active_height && frame.x==left && frame.y==top;
    }
    void ngx_tick(const char *phase) {
      present_started=GetTickCount64();
      step();game3d.no_effects();
      status=game3d.automatic();
      require(query_provider_status(observed.runtime,&provider),"NGX provider UI observation failed");
      unready_presents+=!status.ready();
      if(provider.ready && !provider.reused_depth) {fresh_ended=GetTickCount64();fresh_sequence=provider.current.sequence;}
      trace<<phase<<','<<GetTickCount64()<<','<<++presents<<','<<sequence<<','<<native(ngx_source())<<",present,"<<provider.selected<<','
        <<provider.ready<<','<<provider.reused_depth<<','<<provider.current.resource<<','<<provider.current.sequence<<','<<status.ready()<<','<<status.scale_state<<','<<status.scale<<",,,,,,,\n";
      require(trace.good(),"Cannot write actual NGX trajectory");
    }
    // One native render captured by the production owner; every present in
    // between also runs the caller's per-present check.
    native_render inspect(const char *phase,const std::function<void()> &each={}) {
      const auto frame=game3d.capture(phase,false,[&]{ngx_tick(phase);if(each)each();});
      trace<<phase<<','<<GetTickCount64()<<','<<presents<<','<<sequence<<','<<native(ngx_source())<<",render,,,,,,,,,"
        <<frame.source_resource<<','<<frame.source_id<<','<<frame.depth_ready<<','<<frame.camera_ready<<','<<frame.depth_scale<<','
        <<frame.convergence[1]<<','<<frame.strength_blend<<'\n';
      require(trace.good(),"Cannot write actual NGX trajectory");
      return frame;
    }
    native_render ngx_settle(const char *phase,unsigned timeout=15000) {
      const auto started=GetTickCount64();unsigned continuous=0;
      native_render frame;
      do {
        ngx_tick(phase);
        continuous=present_current() && status.ready() ? continuous+1 : 0;
        if(continuous<6) continue;
        // Sustained readiness is not yet full-strength re-entry; inspect the render.
        frame=inspect(phase);
        if(ngx_current(frame) && frame.camera_ready && frame.strength_blend==1.f && present_current() && status.ready()) {
          std::printf("MEASURE %s elapsed_ms=%llu source=%llu H=%.9g t0=%.9g K=%.9g getters=%u\n",phase,
            static_cast<unsigned long long>(GetTickCount64()-started),static_cast<unsigned long long>(frame.source_id),
            frame.depth_scale,frame.convergence[1],status.scale,ngx_fixture::getters);
          return frame;
        }
        continuous=0;
      } while(GetTickCount64()-started<timeout);
      std::printf("MEASURE failed %s current=%u ready=%u consumed_current=%u camera=%u blend=%.9g provider=%s\n",phase,
        unsigned(present_current()),unsigned(status.ready()),unsigned(ngx_current(frame)),unsigned(frame.camera_ready),
        frame.strength_blend,frame.provider.c_str());
      report_debug_messages();
      throw std::runtime_error("NGX native depth did not reach sustained adaptive rendering");
    }
    void verify_ngx_depth(const native_render &frame) {
      require(frame.camera_ready,"NGX pixel verification requires a current adaptive source");
      verify_ngx_capture(frame);
    }
    void verify_ngx_capture(const native_render &frame) {
      require(ngx_current(frame),"NGX pixel verification requires the current NGX source");
      const auto &source=*consumed_source(frame);
      require(frame.coordinate_basis==1 && !frame.projection_supplied,"NGX without camera metadata invented a projection scale");
      // Native rendering consumes the add-on's completed copy, never the
      // mutable game allocation the SDK received.
      require(frame.captured_resource && frame.captured_resource!=frame.source_resource,
        "Native render consumed mutable NGX game depth instead of its capture");
      // The copy retains the allocation; constants confine sampling to the
      // NGX active rectangle rather than its poisoned padding.
      const std::array<float,4> rect{float(left)/source.width,float(top)/source.height,
        float(active_width)/source.width,float(active_height)/source.height};
      require(frame.depth_rect==rect,"Native render would sample allocation padding instead of the current NGX active rectangle");
      require(frame.raw_depth.size()==size_t(source.width)*source.height*sizeof(float),
        "NGX capture changed allocation dimensions instead of carrying its active rectangle");
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
        const float actual=frame.depth(x,y);
        if(actual!=expected)
          std::printf("MEASURE NGX depth mismatch source=%llu pixel=%u,%u expected=%.9g actual=%.9g\n",
            static_cast<unsigned long long>(native(source)),x,y,expected,actual);
        require(actual==expected,"NGX full allocation copy contains stale or incorrect depth or padding");
      }
      if(use_depth_stencil_crop) {
        require(read_packed_plane(source.resource.p,0,source.state)==frame.raw_depth,"NGX capture modified original packed source depth");
        const auto stencil=read_packed_plane(source.resource.p,1,source.state);
        require(std::all_of(stencil.begin(),stencil.end(),[](auto v){return v==17;}),
          "NGX packed capture modified original source stencil");
      }
    }
    void require_status(bool selected_expected,bool ready_expected) {
      sunshine_streamline::provider::source_status current;
      require(query_provider_status(observed.runtime,&current),"NGX provider UI observation failed");
      if(current.selected!=selected_expected || current.ready!=ready_expected)
        std::printf("MEASURE NGX status mismatch present=%u expected=%u,%u actual=%u,%u provider=%u resource=0x%llx\n",presents,
          unsigned(selected_expected),unsigned(ready_expected),unsigned(current.selected),unsigned(current.ready),
          unsigned(current.provider),static_cast<unsigned long long>(current.current.resource));
      require(current.selected==selected_expected && current.ready==ready_expected,
        "NGX UI selected/ready status disagrees with actual accepted capture ownership");
      if(ready_expected)
        require(current.current.resource==native(ngx_source()) && current.current.identity &&
          current.current.identity==sunshine_native_identity::resource_cookie(ngx_source().resource.p) &&
          current.current.width==active_width && current.current.height==active_height,
          "NGX UI reports the wrong current source or extent");
      else require(!current.current.resource && !current.current.identity,"NGX UI marks historical source as active on a missing current frame");
    }
    // The latest present's exported SBS is the current source color in both eyes.
    std::vector<std::uint8_t> check_exported_mono() { return check_current_mono(game3d.exported()); }
    // This present displays the newest completed copy, unchanged, within its
    // source age bound; a failed, missing or pending newer frame never replaces it.
    bool retained_copy() const {
      return provider.selected && provider.ready && provider.reused_depth && provider.current.sequence==fresh_sequence &&
        present_started<fresh_ended+sunshine_scene_depth::maximum_source_age_ms;
    }
    // Failed, missing or refused input holds the newest completed copy until
    // its source age expires, then renders current-color mono. Calibration and
    // source ownership survive both; only a fresh copy may end the gap.
    void hold_then_mono(const char *phase,const char *message) {
      unsigned held=0,mono=0;
      do {
        ngx_tick(phase);
        require(status.scale==reference_scale && provider.provider==sunshine_scene_depth::provider_kind::ngx,message);
        if(provider.ready) {
          require(!mono && retained_copy() && status.ready(),message);
          ++held;
        } else {
          require(!status.ready(),message);
          require_status(true,false);
          ++mono;
        }
      } while(mono<4);
      check_exported_mono();
      std::printf("MEASURE %s held_presents=%u mono_presents=%u age_bound_ms=%llu\n",phase,held,mono,
        static_cast<unsigned long long>(sunshine_scene_depth::maximum_source_age_ms));
    }
    void require_established(const native_render &frame,const char *message) {
      require(frame.source_id==logical_source && status.scale==reference_scale && frame.depth_scale==reference_scale,message);
    }

    void run_state_pressure() {
      for(unsigned i=0;i<40;++i) pressure_resources.push_back(target_uav(16,16,0));
      std::vector<std::uint8_t> first_output;
      const auto unchanged=[&] {
        require(present_current() && status.ready() && status.scale==reference_scale,
          "Forty unrelated resource states invalidated current NGX depth or its unchanged calibration");
        require_status(true,true);
      };
      state_pressure=true;
      const auto first_present=presents;
      for(unsigned i=0;i<12;++i) {
        selected->pattern=i&1u; // Current pixels change while center scale stays fixed.
        ngx_tick("NGX-source-state-pressure");unchanged();
        if(i==0 || i==11) {
          const auto frame=inspect("NGX-source-state-pressure",unchanged);
          verify_ngx_depth(frame);require_established(frame,"Source-state pressure changed logical source or calibration");
          const auto pixels=game3d.exported();
          if(i==0) first_output=pixels;
          else require(pixels!=first_output,"Native HDR stereo output froze while current NGX depth changed under resource pressure");
        }
      }
      require(pressure_recordings==presents-first_present && pressure_recordings>=12,
        "NGX source-state pressure did not span every genuine command-list reset");
      state_pressure=false;
      const auto recovered=ngx_settle("NGX-state-pressure-reset-recovery");verify_ngx_depth(recovered);
      require_established(recovered,"Ending unrelated state pressure changed the source or retained an invalid recording");
      std::printf("PASS forty unrelated completed split-transition pairs before NGX evaluation preserve exact current 4K HDR depth, export and calibration across %u command-list resets\n",pressure_recordings);
    }
    void run_cross_queue() {
      const unsigned first_submission=producer_submissions;
      const auto first_present=presents;
      std::vector<std::uint8_t> first_output;
      const auto unchanged=[&] {
        require(present_current() && status.ready() && status.scale==reference_scale,
          "Game-ordered separate NGX queue lost current depth or calibration");
        require_status(true,true);
      };
      for(unsigned i=0;i<12;++i) {
        selected->pattern=i&1u;
        ngx_tick("NGX-separate-producer-current-frame");unchanged();
        if(i==0 || i==11) {
          const auto frame=inspect("NGX-separate-producer-current-frame",unchanged);
          verify_ngx_depth(frame);require_established(frame,"Separate NGX queue changed logical source or calibration");
          require(frame.strength_blend==1.f,"Separate NGX queue restarted its stereo strength ramp");
          const auto pixels=game3d.exported();
          if(i==0) first_output=pixels;
          else require(pixels!=first_output,"Separate-queue NGX HDR output froze while producer depth changed");
        }
      }
      require(producer_submissions-first_submission==presents-first_present && presents-first_present>=12,
        "Separate NGX producer did not execute a genuine recording and game queue wait for every present");
      // Each steady present displays its own evaluation; calibrate the
      // provider's sequence numbering against this fixture's evaluations.
      require(present_current(),"Separate NGX queue lost current depth before the replayable-recording case");
      const auto sequence_offset=provider.current.sequence-sequence;
      // A present never displays the copy recorded by its own still-replayable
      // list. Resetting that list for the next frame retires the recording, so
      // its completed copy may then be admitted one presentation later.
      retire_producer_recording=false;
      unsigned lagging=0,held=0,mono=0;
      for(unsigned i=0;i<8;++i) {
        ngx_tick("NGX-completed-producer-still-replayable");
        require(status.scale==reference_scale && (!provider.ready || provider.current.sequence<sequence+sequence_offset),
          "A completed but replayable NGX producer exposed mutable depth to another queue");
        if(provider.ready && !provider.reused_depth) {
          require(provider.current.sequence+1==sequence+sequence_offset && provider.current.resource==native(*selected),
            "Retiring a replayable NGX recording admitted other than its own completed copy");
          ++lagging;
        } else if(provider.ready) {
          require(retained_copy(),"Replayable NGX recording replaced or outlived the newest completed copy");
          ++held;
        } else {require_status(true,false);++mono;}
      }
      std::printf("MEASURE NGX replayable recordings retired_predecessor=%u held=%u mono=%u\n",lagging,held,mono);
      retire_producer_recording=true;
      const auto retired=ngx_settle("NGX-retired-producer-recording-recovery");verify_ngx_depth(retired);
      require_established(retired,"Retiring the replayable NGX recording changed source or calibration");
      std::puts("PASS a still-replayable producer recording is never displayed; its retired predecessor and a fully retired recording supply immutable captures without recalibration");
      check_pending_cross_queue();
      emit=false;
      hold_then_mono("NGX-separate-producer-missing-evaluation","Separate producer queue exposed historical depth after a missing NGX evaluation");
      emit=true;
      const auto recovered=ngx_settle("NGX-separate-producer-gap-recovery");verify_ngx_depth(recovered);
      // Repeated real queue lifetimes catch retained queue/fence ownership. The
      // main presentation queue stays alive; replacing it needs a new swapchain
      // and is a different lifecycle from the producer that owns NGX work.
      for(unsigned i=0;i<3;++i) {
        create_producer_queue();
        selected->pattern=i&1u;
        const auto frame=ngx_settle("NGX-producer-queue-recreation");verify_ngx_depth(frame);require_status(true,true);
        require_established(frame,"NGX producer queue recreation changed logical source or calibration");
      }
      require(producer_generation==4,"NGX producer queue retirement was not exercised across four native queue lifetimes");
      std::puts("PASS game-ordered separate NGX producer queue drives exact current 4K HDR depth and native output across resets, missing-frame mono/recovery and four producer queue lifetimes");
    }
    void check_pending_cross_queue() {
      selected->pattern^=1u;
      com_ptr<ID3D12Fence> gate;
      checked(game->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(gate.put())),"Create pending producer admission gate");
      HANDLE release_observed=CreateEventW(nullptr,TRUE,FALSE,nullptr);
      require(release_observed,"Create pending producer release event");
      bool observed_pending=false,producer_unfinished=false,published_fresh=true,published_other=true;
      const auto completed_sequence=fresh_sequence;
      HRESULT release_result=E_FAIL;
      std::atomic<bool> emergency_release{false};
      // Release the GAME's gate only after native rendering has declined the
      // unfinished foreign copy. It may hold only the newest completed copy.
      // A CPU completion wait requires the watchdog and fails the test; no
      // add-on GPU dependency should be necessary.
      ngx_frame_observer=[&] {
        if(observed_pending) return;
        observed_pending=true;
        const auto completed=producer_completion->GetCompletedValue();
        producer_unfinished=completed!=UINT64_MAX && completed<producer_fence_value;
        // This callback runs inside the runtime; record, never throw.
        sunshine_streamline::provider::source_status current;
        if(query_provider_status(observed.runtime,&current)) {
          published_fresh=current.ready && !current.reused_depth;
          published_other=current.ready && current.reused_depth && current.current.sequence!=completed_sequence;
        }
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
        checked(producer_queue->Wait(gate.p,1),"Hold actual NGX producer pending until native rendering observes it");
        ngx_tick("NGX-separate-producer-pending-gate");
      } catch(...) {
        gate->Signal(1);SetEvent(release_observed);release_watchdog.join();
        ngx_frame_observer={};producer_gated=false;CloseHandle(release_observed);
        throw;
      }
      release_watchdog.join();ngx_frame_observer={};producer_gated=false;CloseHandle(release_observed);
      require(!emergency_release && observed_pending && SUCCEEDED(release_result),
        "Pending NGX producer blocked the CPU before native rendering");
      require(producer_unfinished && !published_fresh && !published_other,
        "Unfinished foreign NGX producer exposed its depth pixels");
      if(provider.ready) require(retained_copy() && status.ready(),"Pending NGX producer replaced or outlived the newest completed copy");
      else {require(!status.ready(),"Pending NGX producer rendered stereo without depth");require_status(true,false);check_exported_mono();}
      require(provider.provider==sunshine_scene_depth::provider_kind::ngx && status.scale==reference_scale,
        "Pending NGX producer lost source authority or estimated calibration");
      const auto recovered=ngx_settle("NGX-separate-producer-pending-recovery");verify_ngx_depth(recovered);
      require_established(recovered,"Completed NGX recovery changed logical source or calibration");
      std::printf("PASS unfinished foreign NGX producer is never displayed (held_completed_copy=%u) without a CPU wait, preserves source/calibration, and recovers exact depth and stereo after completion\n",
        unsigned(provider.ready));
    }
    void run_cross_queue_async() {
      // Before the first accepted NGX copy, Generic legitimately still owns
      // depth. Establish ownership using real completed/retired producer work,
      // then remove only the fixture's CPU completion wait for this phase.
      complete_producer=true;
      const auto warm=ngx_settle("NGX-async-established-owner-warmup");verify_ngx_depth(warm);require_status(true,true);
      logical_source=warm.source_id;reference_scale=status.scale;
      require(logical_source && status.active_scale() && warm.depth_scale==reference_scale,
        "Asynchronous NGX regression did not first establish its own adaptive source");
      complete_producer=false;
      unsigned captures=0,held=0,camera_frames=0,mono_frames=0,verified=0;
      const auto classify=[&] {
        if(provider.ready && !provider.reused_depth) {
          require(present_current(),"Asynchronous NGX producer exposed a previous source, Generic fallback or invalid current extent");
          ++captures;require_status(true,true);
        } else if(provider.ready) {
          require(retained_copy(),"Asynchronous NGX hold replaced or outlived its newest completed copy");
          ++held;
        } else require_status(true,false);
        require(status.scale==reference_scale,"Asynchronous NGX admission changed the established depth calibration");
        if(status.ready()) {
          require(provider.ready,"Asynchronous NGX camera rendered without an admitted copy");
          ++camera_frames;
        } else ++mono_frames;
      };
      for(unsigned i=0;i<32;++i) {
        selected->pattern=i&1u;
        ngx_tick("NGX-separate-producer-async-admission");classify();
        if(!status.ready() && (i==0 || i==31)) check_exported_mono();
        if(status.ready() && verified<2) {
          // Hold this pattern while the production owner captures one render.
          const auto frame=inspect("NGX-separate-producer-async-admission",classify);
          if(frame.depth_ready) require(frame.source_id==logical_source && (frame.reused_depth || ngx_current(frame)),
            "Asynchronous NGX render consumed a previous source or Generic fallback");
          if(frame.camera_ready && !frame.reused_depth) {verify_ngx_depth(frame);++verified;}
        }
      }
      std::printf("PASS asynchronous game-ordered NGX queue admitted only current sources or held the newest completed copy: captures=%u held=%u stereo=%u mono=%u exact_renders=%u; readiness is conservatively optional until the producer completes\n",
        captures,held,camera_frames,mono_frames,verified);
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
      create_pipeline();create_ngx_compute();
      active_width=width==3840 ? 2227 : width/2-1;active_height=height==2160 ? 1253 : height/2-1;
      left=3;top=2;
      first=target_uav(active_width+7,active_height+5,0);
      second=target_uav(active_width+7,active_height+5,1);selected=first.get();
      depth_stencil_crop=target_packed(active_width+7,active_height+5,0);
      decoy=target(width,height,1,9,false,true);
      valid_evaluation=false;
      cross_queue=sunshine_camera_fixture::flag("SUNSHINE_NGX_CROSS_QUEUE_TEST");
      complete_producer=sunshine_camera_fixture::flag("SUNSHINE_NGX_CROSS_QUEUE_COMPLETED_TEST");
      require(!complete_producer || cross_queue,"Completed NGX producer option requires the cross-queue test");
      require(!cross_queue || !sunshine_camera_fixture::flag("SUNSHINE_NGX_STATE_PRESSURE_TEST"),
        "Run distinct NGX queue and state-pressure regressions separately");
      if(cross_queue) create_producer_queue();
      trace.open(runtime_directory/"ngx-depth-trajectory.csv");
      trace<<std::setprecision(17)<<"phase,wall_ms,present,evaluation,tagged,kind,selected,capture_ready,reused,current_resource,current_sequence,automatic_ready,scale_state,K,"
        "consumed_source,source_id,consumed_ready,camera_ready,H,t0,blend\n";
      reshade::register_event<reshade::addon_event::reset_command_list>(observe_reset);
      reshade::register_event<reshade::addon_event::finish_present>(observe_ngx_present);
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
      query_provider_status=reinterpret_cast<provider_status_t>(GetProcAddress(module,"SunshineDepthTestProviderStatus"));
      require(manual && query_provider_status,"NGX fixture requires passive provider/UI observation adapters");
      game3d.start();game3d.attach_export();game3d.set_strength(100);

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
        // The evaluating recording observes no state for its source. NGX
        // inputs then use the DLSS input-state contract; the copy must still
        // contain this frame's exact depth, never stale or Generic depth.
        const auto admitted=[&] {
          require(present_current(),"A valid NGX evaluation without observed source state was not admitted by its input contract");
          require_status(true,true);
        };
        evaluate_without_source_state=true;
        ngx_tick("first-valid-NGX-without-native-state");admitted();
        const auto frame=inspect("first-valid-NGX-without-native-state",admitted);
        verify_ngx_capture(frame);
        evaluate_without_source_state=false;
        std::puts("PASS successful valid NGX evaluation with no observed source state is admitted through the DLSS input-state contract with exact current depth");
      }
      const auto finish=[&] {
        render_tracked_depth={};
        reshade::unregister_event<reshade::addon_event::finish_present>(observe_ngx_present);
        reshade::unregister_event<reshade::addon_event::reset_command_list>(observe_reset);
      };
      if(cross_queue && !complete_producer) {
        run_cross_queue_async();
        finish();
        return;
      }
      const auto established=ngx_settle(cross_queue ? "NGX-completed-separate-producer" : "ngx-low-padded-UAV");
      verify_ngx_depth(established);require_status(true,true);
      for(unsigned i=0;i<4;++i) {
        ngx_tick("NGX-stable-native-ownership");require_status(true,true);
        require(status.ready(),"Steady native NGX frames interrupted Automatic readiness");
      }
      require(!manual(observed.runtime,native(*selected)),"NGX-only non-DSV source unexpectedly entered Generic inventory");
      require(status.basis==unsigned(sunshine_game3d::automatic_scale_basis::relative_depth) && status.active_scale() &&
        established.depth_scale==status.scale,
        "NGX depth without projection did not initialize an active relative-depth scale applied by native rendering");
      logical_source=established.source_id;reference_scale=status.scale;
      require(logical_source && established.provider_source_id==logical_source,"NGX logical depth source identity is missing");
      std::printf("MEASURE NGX established reference H=%.9g t0=%.9g\n",established.depth_scale,established.convergence[1]);
      std::puts("PASS real lower-resolution NGX UAV with poisoned padded extent drives correct HDR adaptive native rendering without Generic inventory");

      if(cross_queue || sunshine_camera_fixture::flag("SUNSHINE_NGX_STATE_PRESSURE_TEST")) {
        if(cross_queue) run_cross_queue();else run_state_pressure();
        finish();
        return;
      }

      // A first tag identifies this resource only after its earlier transition.
      // Its evaluation still follows the DLSS input-state contract, so the
      // first sighting is current and keeps the unchanged encoding's scale.
      selected=second.get();
      ngx_tick("new-NGX-resource-first-sighting");
      std::printf("MEASURE NGX first sighting capture_ready=%u automatic_ready=%u scale_state=%u K=%.9g\n",
        unsigned(provider.ready),unsigned(status.ready()),status.scale_state,status.scale);
      require(present_current() && status.scale==reference_scale,
        "First sighting of a new NGX resource was not admitted or recalibrated the unchanged encoding");
      require_status(true,true);
      const auto sighted=ngx_settle("new-NGX-resource-first-sighting");verify_ngx_depth(sighted);
      require_established(sighted,"First sighting of a new NGX resource recalibrated the unchanged NGX encoding");
      std::puts("PASS first sighting of a new NGX resource is admitted by its input contract and retains logical calibration");
      rotate=true;ngx_settle("NGX-rotation-warmup");unsigned seen=0,verified=0;
      const auto rotating=[&] {
        require(present_current() && status.ready() && status.scale==reference_scale,
          "NGX rotation reset calibration or interrupted current capture");
        require_status(true,true);seen|=selected==first.get()?1u:2u;
      };
      for(unsigned i=0;i<16;++i) {
        ngx_tick("rotating-NGX-resources");rotating();
        if(i<2) {
          const auto frame=inspect("rotating-NGX-resources",rotating);
          verify_ngx_depth(frame);require_established(frame,"NGX rotation changed its logical feature identity or calibration");
          verified|=frame.source_resource==native(*first)?1u:2u;
        }
      }
      require(seen==3,"NGX rotation did not capture both physical resources");
      std::printf("MEASURE NGX rotation exact_render_members=%u\n",verified);
      rotate=false;selected=first.get();
      auto replacement=target_uav(first->width,first->height,0);
      first=std::move(replacement);selected=first.get();
      const auto recreated=ngx_settle("resource-recreation");verify_ngx_depth(recreated);
      require_established(recreated,"NGX physical resource recreation changed the stable logical depth domain");
      std::puts("PASS NGX physical rotation and recreation retain logical source and calibration");

      use_depth_stencil_crop=true;
      const auto packed=ngx_settle("packed-NGX-padded-depth");verify_ngx_depth(packed);require_status(true,true);
      require_established(packed,"Packed NGX depth allocation changed the logical raw encoding or scale");
      std::puts("PASS padded packed D32S8 depth uses legal whole-plane capture, exact sampling rectangle and unchanged original depth/stencil");
      use_depth_stencil_crop=false;
      verify_ngx_depth(ngx_settle("packed-to-UAV-NGX-return"));

      valid_evaluation=false;
      hold_then_mono("failed-established-evaluation","Failed NGX frames replaced or outlived the retained depth");
      valid_evaluation=true;ngx_settle("failed-evaluation-recovery");
      parameters.provide_depth=false;
      hold_then_mono("missing-established-depth","NGX without depth fell back to an unrelated Generic scene or outlived the retained depth");
      parameters.provide_depth=true;ngx_settle("missing-depth-recovery");
      emit=false;
      const auto silence_began=GetTickCount64();
      hold_then_mono("silent-established-provider","Silent NGX provider exposed stale depth or Generic fallback");
      // A silent provider keeps the queue mono only for the association
      // timeout (one second without an evaluation); then Generic may run.
      bool association_released=false;
      do {
        ngx_tick("silent-established-provider");
        sunshine_streamline::provider::source_status current;
        require(query_provider_status(observed.runtime,&current),"NGX provider UI observation failed");
        require(!provider.ready && !current.ready,"Silent NGX provider exposed stale depth");
        if(GetTickCount64()-silence_began<900)
          require(current.selected && !status.ready() && status.scale==reference_scale,
            "Silent NGX provider released its association before the timeout");
        // Released, the fixture's Generic scene depth may take over.
        if(!current.selected) association_released=true;
        else require(!association_released,"A released NGX association returned without an evaluation");
      } while(GetTickCount64()-silence_began<1800);
      require(association_released,"Silent NGX provider kept the queue mono past the association timeout");
      emit=true;
      const auto initial=ngx_settle("silent-provider-recovery");
      require_established(initial,"NGX temporary gap changed its logical depth domain or calibration");
      std::puts("PASS established NGX failures and missing depth keep source ownership, hold the newest completed copy within its age bound, then render current-color mono; one second of silence releases the association; all recover");

      // A center change starts smooth refinement with its fresh measurements.
      // Refinement never interrupts the current source or its readiness.
      center_raw=.03125f;
      const auto adapt_until=GetTickCount64()+5000;
      do {
        ngx_tick("zero-plane-NGX-reference");
        require(present_current() && status.ready() && status.active_scale(),
          "NGX scene refinement changed source or interrupted readiness");
        require_status(true,true);
      } while(GetTickCount64()<adapt_until);
      const auto refined=inspect("zero-plane-NGX-reference");verify_ngx_depth(refined);
      require(refined.source_id==logical_source && refined.strength_blend==1.f && refined.depth_scale==status.scale,
        "NGX scene refinement changed its logical source, stereo re-entry or applied scale");
      require(refined.convergence[1]>initial.convergence[1],"NGX screen-plane reference did not follow the nearer current center");
      std::printf("MEASURE NGX screen-plane refinement initial H=%.9g t0=%.9g final H=%.9g t0=%.9g\n",
        initial.depth_scale,initial.convergence[1],refined.depth_scale,refined.convergence[1]);
      std::puts("PASS NGX reference refinement keeps the current source and full-strength readiness");

      // Recalibration starts a fresh reference from current actual depth; two
      // fresh references of one unchanged scene are identical.
      selected->pattern=2;
      const auto recalibrated=[&](const char *phase) {
        const auto before=unready_presents;
        require(game3d.recalibrate(),"NGX explicit recalibrate action was rejected");
        const auto frame=ngx_settle(phase);verify_ngx_depth(frame);
        require(unready_presents>before && frame.source_id==logical_source,
          "NGX recalibration reused its previous reference or changed logical source");
        return frame;
      };
      const auto fresh=recalibrated("constant-NGX-background");
      const auto repeated=recalibrated("constant-NGX-background-repeat");
      std::printf("MEASURE NGX recalibrated reference H=%.9g t0=%.9g repeat H=%.9g t0=%.9g\n",
        fresh.depth_scale,fresh.convergence[1],repeated.depth_scale,repeated.convergence[1]);
      require(repeated.depth_scale==fresh.depth_scale && repeated.convergence==fresh.convergence,
        "NGX recalibration did not initialize a reproducible fresh reference from current depth");
      game3d.set_strength(0);for(unsigned i=0;i<4;++i)ngx_tick("strength-zero");
      const auto mono_pixels=check_exported_mono();
      game3d.set_strength(50);ngx_settle("strength-half");
      const auto half_shift=shifts(mono_pixels,game3d.exported(),"NGX-50");
      game3d.set_strength(100);ngx_settle("strength-full");
      const auto full_pixels=game3d.exported();
      const auto full_shift=shifts(mono_pixels,full_pixels,"NGX-100");
      for(unsigned eye=0;eye<2;++eye) {
        require(std::abs(half_shift[eye])>=.5f && std::abs(full_shift[eye])>std::abs(half_shift[eye])+.5f &&
          std::abs(full_shift[eye]-2.f*half_shift[eye])<=1.f,"NGX actual HDR stereo does not respond proportionally to linear user strength");
      }
      sunshine_parity::write_bytes(runtime_directory/"ngx-final.sbs",full_pixels.data(),full_pixels.size());
      std::puts("PASS actual NGX HDR output preserves zero-strength color and restores proportional 0/50/100 strength");

      require(call_release(feature)==ngx_fixture::success,"NGX feature release failed");
      feature=nullptr;auto_create=false;
      const auto released_status=[&]{require_status(false,false);};
      for(unsigned i=0;i<4;++i) {ngx_tick("explicit-NGX-release");released_status();}
      const auto released=inspect("explicit-NGX-release",released_status);
      require(!released.depth_ready || (released.source_resource!=native(*selected) && released.provider!="ngx"),
        "Released NGX feature still supplied historical depth as current");
      std::puts("PASS explicit NGX feature release returns source ownership to Generic; temporary missing frames remain a distinct held state");
      auto_create=true;reversed=false;selected->pattern=0;center_raw=.015625f;
      const auto before_new=unready_presents;
      const auto renewed=ngx_settle("normal-depth-new-feature");verify_ngx_depth(renewed);
      std::printf("MEASURE NGX normal-depth feature H=%.9g t0=%.9g orientation=%u\n",renewed.depth_scale,renewed.convergence[1],renewed.detected_orientation);
      require(renewed.source_id!=logical_source && unready_presents>before_new,
        "New NGX feature with a different depth convention inherited the old source calibration");
      require(renewed.depth_scale==established.depth_scale && renewed.convergence==established.convergence,
        "Normal-depth NGX scene did not reproduce the reversed-depth reference for the same nearness");
      require(ngx_fixture::created==2 && ngx_fixture::released==1 && ngx_fixture::evaluated==sequence,
        "NGX production interception changed native call counts");
      std::puts("PASS NGX feature recreation carries a new logical domain and supports normal depth without camera projection");
      finish();
      std::puts("PASS production NGX SDK discovery, native copy/submission, adaptive scale, current source UI and native HDR rendering; no FX");
    }
  };
}

int main(int argc,char **argv) {
  std::setvbuf(stdout,nullptr,_IONBF,0);
  if(argc!=5 && argc!=7) {std::fputs("usage: ngx_depth_runtime <official.dll> <Shaders> <test.addon64> <fresh-output> [width height]\n",stderr);return 2;}
  std::thread([]{Sleep(300000);std::fputs("FAIL NGX runtime watchdog\n",stderr);TerminateProcess(GetCurrentProcess(),124);}).detach();
  try {
    // These cases observed FX callbacks and uniforms. Native SL/FG coverage
    // belongs to reshade_game3d_native_provider_runtime_test.
    for(const char *retired : {"SUNSHINE_NGX_FRAME_GENERATION_TEST","SUNSHINE_SL_TAG_FALLBACK_TEST","SUNSHINE_FG_LIVE_COMPAT_TEST",
        "SUNSHINE_FG_LATE_SOURCE_TEST","SUNSHINE_FG_SUSTAINED_PRODUCER_TEST","SUNSHINE_FG_PENDING_PRODUCER_TEST",
        "SUNSHINE_FG_PRODUCER_ORACLE_ONLY_TEST","SUNSHINE_FG_SKIP_NGX_GATED_TEST","SUNSHINE_FG_CONTIGUOUS_COPY_TEST",
        "SUNSHINE_FG_IDENTITY_PRECISION_TEST","SUNSHINE_FG_SCALE_UI_TEST","SUNSHINE_FG_PUBLICATION_GAP_TEST",
        "SUNSHINE_FG_FRESH_RETIRE_ALLOCATOR_TEST","SUNSHINE_EXPECT_LEGACY_MATRIX_GAIN","SUNSHINE_NGX_TRACKED_SOURCE_TEST",
        "SUNSHINE_CAPTURE_DEMAND_TEST","SUNSHINE_NGX_PENDING_CONTINUITY_TEST","SUNSHINE_DEPTH_BIND_SWITCH_TEST"})
      if(sunshine_camera_fixture::flag(retired)) {
        std::fprintf(stderr,"SETUP %s selects a retired effect-based NGX case; use reshade_game3d_native_provider_runtime_test for native SL/FG\n",retired);
        return 2;
      }
    width=argc==7?unsigned(std::stoul(argv[5])):3840;height=argc==7?unsigned(std::stoul(argv[6])):2160;
    require(width>=640 && width<=3840 && height>=360 && height<=2160 && width%4==0 && height%4==0,"Invalid NGX fixture dimensions");
    select_native_boot();
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
    ngx_runtime_fixture fixture;fixture.runtime_directory=fs::absolute(argv[4]);
    fixture.initialize(fs::absolute(argv[1]),fs::absolute(argv[2]),fixture.runtime_directory,2,0,fs::absolute(argv[3]));
    if(sunshine_camera_fixture::flag("SUNSHINE_D3D12_DEBUG_TEST"))
      checked(fixture.game->QueryInterface(IID_PPV_ARGS(fixture.debug_messages.put())),"Requested D3D12 debug layer did not expose its diagnostic queue");
    fixture.run_ngx();return 0;
  } catch(const std::exception &error) {std::fprintf(stderr,"FAIL %s\n",error.what());return 1;}
}
