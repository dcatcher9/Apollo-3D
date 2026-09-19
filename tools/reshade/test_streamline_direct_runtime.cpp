// SPDX-License-Identifier: GPL-3.0-only
// Real UAV scene depth -> native middleware copy/submission -> projection
// controller -> full Automatic HDR shader. Only immutable middleware metadata
// is synthetic; no depth binding, readiness, sampler result or fence is injected.
#include "test_raw_runtime_fixture.h"
#include "depth_addon.h"
#include "streamline_depth_provider.h"
#include "native_resource_identity.h"

namespace {
  using capture_t = std::uint64_t (*)(std::uint64_t, std::uint64_t, std::uint32_t,
    const sunshine_streamline::camera_data *, std::uint64_t, bool);
  using finish_capture_t = void (*)(std::uint64_t, bool);
  using provider_status_t = BOOL (*)(api::effect_runtime *, sunshine_streamline::provider::source_status *);
  using frame_t = BOOL (*)(api::effect_runtime *, sunshine_depth::frame_depth *);
  using select_t = BOOL (*)(api::effect_runtime *, std::uint64_t);
  using action_t = BOOL (*)(api::effect_runtime *);
  frame_t query_frame{};
  sunshine_depth::frame_depth captured;
  unsigned captured_render{};
  std::uint64_t game_native_command{};
  bool captured_ready{};
  constexpr GUID streamline_v1_state_guid{0x694b3e1c,0x0e33,0x416f,{0xba,0x83,0xfe,0x24,0x8d,0xa1,0xe8,0x5d}};
  constexpr std::uint32_t chi_shader_read=(1u<<5)|(1u<<6), chi_present=1u<<18;
  void observe_reset(api::command_list *commands) { game_native_command = commands->get_native(); }
  void observe_source(api::effect_runtime *runtime, api::effect_technique technique,
      api::command_list *, api::resource_view, api::resource_view) {
    char name[256] {}; runtime->get_technique_name(technique, name);
    if (!named(name, technique_name)) return;
    captured = {};
    captured_ready = query_frame && query_frame(runtime, &captured) && captured.ready;
    captured_render = observed.renders;
  }

  struct direct_fixture : raw_runtime_fixture {
    struct uav_target {
      com_ptr<ID3D12Resource> resource;
      com_ptr<ID3D12DescriptorHeap> descriptors;
      unsigned width{}, height{}, pattern{};
      D3D12_RESOURCE_STATES state{D3D12_RESOURCE_STATE_UNORDERED_ACCESS};
      std::uint8_t stencil_seed{};
    };
    std::unique_ptr<uav_target> first, second;
    std::unique_ptr<uav_target> barrier_scratch;
    uav_target *selected{};
    com_ptr<ID3D12RootSignature> compute_root;
    com_ptr<ID3D12PipelineState> compute_pipeline;
    sunshine_streamline::camera_data camera;
    capture_t capture{}, capture_v1{}, begin_capture_v1{};
    finish_capture_t finish_capture{};
    provider_status_t query_provider_status{};
    sunshine_streamline::provider::source_description last_ui_source;
    enum class capture_mode { version_two, v1_observed_state, v1_unknown_state, v1_common_state };
    capture_mode mode{capture_mode::version_two};
    enum class private_case { shader_read, shader_read_present, common, missing, short_value, unsupported, conflicting, split };
    private_case private_input{private_case::shader_read};
    bool private_state_mode{}, packed_state_mode{};
    bool defer_evaluation_finish{}, replay_evaluation_recording{};
    unsigned submitted_before_finish{}, replayed_recordings{};
    com_ptr<ID3D12CommandQueue> batch_queue;
    std::vector<ID3D12CommandList *> batch_idle;
    bool large_submission{}, idle_only_after_submission{}, large_barriers{};
    unsigned batch_producer_index{}, submitted_large_batches{}, submitted_idle_batches{}, recorded_large_barriers{};
    std::array<bool,2> packed_layout_logged{};
    select_t manual{};
    // Compatibility field name shared with the derived NGX fixture; the action
    // now explicitly resets both the reference and the zero plane.
    action_t recenter{};
    std::ofstream trace;
    std::uint64_t sequence{}, ticket{};
    bool emit = true, rotate = false, valid_evaluation = true;
    unsigned rotation_index{};
    unsigned v1_capture_calls{}, v2_capture_calls{};
    float center_raw = .015625f;
    // Capture-only variants intentionally vary the center during startup. The
    // functional reference/matrix scenarios instead use an independent oracle.
    float expected_reference = std::numeric_limits<float>::quiet_NaN();

    static std::uint64_t native(const uav_target &v) { return reinterpret_cast<std::uint64_t>(v.resource.p); }
    void initialize_camera() {
      camera.near_plane = .0625f;
      camera.far_plane = std::numeric_limits<float>::infinity();
      camera.fov = 1.1f; camera.aspect = float(width)/height;
      const float sy = 1.f/std::tan(camera.fov*.5f), sx = sy/camera.aspect;
      camera.projection.m[0][0] = sx; camera.projection.m[1][1] = sy;
      camera.projection.m[2][3] = 1.f; camera.projection.m[3][2] = camera.near_plane;
      camera.inverse_projection.m[0][0] = 1.f/sx; camera.inverse_projection.m[1][1] = 1.f/sy;
      camera.inverse_projection.m[2][3] = 1.f/camera.near_plane;
      camera.inverse_projection.m[3][2] = 1.f;
      camera.depth_inverted = 1; camera.orthographic = camera.reset = camera.not_rendering_game_frames = 0;
    }
    void create_compute() {
      const char *source = R"(
cbuffer Settings : register(b0) { uint w; uint h; uint pattern; float center_raw; };
RWTexture2D<float> depth : register(u0);
[numthreads(8,8,1)] void main(uint3 id : SV_DispatchThreadID) {
  if (id.x >= w || id.y >= h) return;
  float2 uv = (float2(id.xy)+.5)/float2(w,h);
  uint2 cell = min(uint2(31,17),uint2(uv*float2(32,18)));
  bool center = cell.x >= 14 && cell.x < 18 && cell.y >= 7 && cell.y < 11;
  float d = pattern == 2 ? .015625 : .0078125*(float((pattern ? 31-cell.x : cell.x)/8)+1);
  depth[id.xy] = center ? center_raw : d;
})";
      com_ptr<ID3DBlob> shader, errors, signature;
      checked(D3DCompile(source, std::strlen(source), nullptr, nullptr, nullptr, "main", "cs_5_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, shader.put(), errors.put()), "Compile native UAV depth producer");
      D3D12_DESCRIPTOR_RANGE range{};
      range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; range.NumDescriptors = 1;
      D3D12_ROOT_PARAMETER parameters[2]{};
      parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
      parameters[0].Constants.Num32BitValues = 4;
      parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
      parameters[1].DescriptorTable = {1,&range};
      D3D12_ROOT_SIGNATURE_DESC desc{};
      desc.NumParameters = 2; desc.pParameters = parameters;
      checked(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, signature.put(), errors.put()),
        "Serialize native UAV depth root signature");
      checked(game->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
        IID_PPV_ARGS(compute_root.put())), "Create native UAV depth root signature");
      D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline{};
      pipeline.pRootSignature = compute_root.p;
      pipeline.CS = {shader->GetBufferPointer(),shader->GetBufferSize()};
      checked(game->CreateComputePipelineState(&pipeline,IID_PPV_ARGS(compute_pipeline.put())), "Create native UAV depth pipeline");
    }
    std::unique_ptr<uav_target> target_uav(unsigned w,unsigned h,unsigned pattern) {
      auto result = std::make_unique<uav_target>();
      result->width=w; result->height=h; result->pattern=pattern;
      const auto heap = heap_properties(D3D12_HEAP_TYPE_DEFAULT);
      D3D12_RESOURCE_DESC desc{};
      desc.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D; desc.Width=w; desc.Height=h;
      desc.DepthOrArraySize=desc.MipLevels=desc.SampleDesc.Count=1;
      desc.Format=DXGI_FORMAT_R32_FLOAT; desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
      checked(game->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,IID_PPV_ARGS(result->resource.put())), "Create non-DSV tagged depth UAV");
      D3D12_DESCRIPTOR_HEAP_DESC descriptors{};
      descriptors.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; descriptors.NumDescriptors=1;
      descriptors.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
      checked(game->CreateDescriptorHeap(&descriptors,IID_PPV_ARGS(result->descriptors.put())), "Create depth UAV descriptor heap");
      D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
      view.Format=DXGI_FORMAT_R32_FLOAT; view.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE2D;
      game->CreateUnorderedAccessView(result->resource.p,nullptr,&view,result->descriptors->GetCPUDescriptorHandleForHeapStart());
      return result;
    }
    std::unique_ptr<uav_target> target_packed(unsigned w,unsigned h,unsigned pattern) {
      auto result=std::make_unique<uav_target>();
      result->width=w; result->height=h; result->pattern=pattern;
      result->state=D3D12_RESOURCE_STATE_DEPTH_WRITE;
      const auto heap=heap_properties(D3D12_HEAP_TYPE_DEFAULT);
      D3D12_RESOURCE_DESC desc{};
      desc.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D; desc.Width=w; desc.Height=h;
      desc.DepthOrArraySize=desc.MipLevels=desc.SampleDesc.Count=1;
      desc.Format=DXGI_FORMAT_R32G8X24_TYPELESS; desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
      D3D12_CLEAR_VALUE clear{}; clear.Format=DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
      checked(game->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,result->state,
        &clear,IID_PPV_ARGS(result->resource.put())),"Create real packed D32S8 tagged source");
      D3D12_DESCRIPTOR_HEAP_DESC descriptors{};
      descriptors.Type=D3D12_DESCRIPTOR_HEAP_TYPE_DSV; descriptors.NumDescriptors=1;
      checked(game->CreateDescriptorHeap(&descriptors,IID_PPV_ARGS(result->descriptors.put())),"Create packed source DSV heap");
      D3D12_DEPTH_STENCIL_VIEW_DESC view{};
      view.Format=clear.Format; view.ViewDimension=D3D12_DSV_DIMENSION_TEXTURE2D;
      game->CreateDepthStencilView(result->resource.p,&view,result->descriptors->GetCPUDescriptorHandleForHeapStart());
      return result;
    }
    void write_packed_pixels() {
      const auto dsv=selected->descriptors->GetCPUDescriptorHandleForHeapStart();
      selected->stencil_seed=static_cast<std::uint8_t>(17+sequence%31+selected->pattern*64);
      // These sources have no scene draws or generic nomination requirement.
      // Real DSV clears establish exact depth and independently changing stencil.
      for(unsigned band=0;band<4;++band) {
        const D3D12_RECT rect{LONG(band*selected->width/4),0,LONG((band+1)*selected->width/4),LONG(selected->height)};
        const float depth=.0078125f*float((selected->pattern ? 3-band : band)+1);
        commands->ClearDepthStencilView(dsv,D3D12_CLEAR_FLAG_DEPTH|D3D12_CLEAR_FLAG_STENCIL,
          depth,static_cast<UINT8>(selected->stencil_seed+band),1,&rect);
      }
      const D3D12_RECT center{LONG(14*selected->width/32),LONG(7*selected->height/18),
        LONG(18*selected->width/32),LONG(11*selected->height/18)};
      commands->ClearDepthStencilView(dsv,D3D12_CLEAR_FLAG_DEPTH,center_raw,0,1,&center);
    }
    void write_depth_pixels() {
      ID3D12DescriptorHeap *heaps[]{selected->descriptors.p};
      commands->SetDescriptorHeaps(1,heaps);
      commands->SetComputeRootSignature(compute_root.p);
      commands->SetPipelineState(compute_pipeline.p);
      struct { unsigned w,h,pattern; float center; } constants{selected->width,selected->height,selected->pattern,center_raw};
      commands->SetComputeRoot32BitConstants(0,4,&constants,0);
      commands->SetComputeRootDescriptorTable(1,selected->descriptors->GetGPUDescriptorHandleForHeapStart());
      commands->Dispatch((selected->width+7)/8,(selected->height+7)/8,1);
      D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV;
      barrier.UAV.pResource=selected->resource.p; commands->ResourceBarrier(1,&barrier);
    }
    void draw_private_frame() {
      if (rotate) selected = (++rotation_index&1u) ? first.get() : second.get();
      draw(*decoy);
      if(selected->state!=D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        transition(commands.p,selected->resource.p,selected->state,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      center_raw=.015625f+float(sequence%8)*.0009765625f;
      write_depth_pixels();
      const auto established=private_input==private_case::common ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
      transition(commands.p,selected->resource.p,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,established);
      selected->state=established;
      // Finish the producer recording before evaluation. Its transition is not
      // proof for the freshly reset evaluation recording; only the provider's
      // exact private metadata describes the source state in the positive cases.
      submit(false);
      begin_commands();
      const auto native_command=game_native_command;
      D3D12_RESOURCE_BARRIER pending{};
      pending.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      pending.Transition={selected->resource.p,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
        established,D3D12_RESOURCE_STATE_UNORDERED_ACCESS};
      if(private_input==private_case::conflicting) {
        commands->ResourceBarrier(1,&pending);
        selected->state=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      } else if(private_input==private_case::split) {
        pending.Flags=D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY;
        commands->ResourceBarrier(1,&pending);
      }
      std::uint32_t encoded=chi_shader_read;
      if(private_input==private_case::shader_read_present) encoded|=chi_present;
      else if(private_input==private_case::common) encoded=chi_present;
      else if(private_input==private_case::unsupported) encoded=0x80000000u;
      if(private_input==private_case::missing)
        checked(selected->resource->SetPrivateData(streamline_v1_state_guid,0,nullptr),"Remove provider-private state");
      else if(private_input==private_case::short_value) {
        const std::uint16_t short_value=static_cast<std::uint16_t>(encoded);
        checked(selected->resource->SetPrivateData(streamline_v1_state_guid,sizeof(short_value),&short_value),"Write malformed provider-private state size");
      } else
        checked(selected->resource->SetPrivateData(streamline_v1_state_guid,sizeof(encoded),&encoded),"Write provider-private chi state");
      ticket=0;
      if(capture_v1 && emit) {
        require(native_command,"Private-state evaluation lost its actual native command identity");
        ++v1_capture_calls;
        ticket=capture_v1(native_command,native(*selected),0,&camera,++sequence,valid_evaluation);
      }
      if(private_input==private_case::split) {
        pending.Flags=D3D12_RESOURCE_BARRIER_FLAG_END_ONLY;
        commands->ResourceBarrier(1,&pending);
        selected->state=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      }
      // A nonzero ticket records source nomination, even when its state does
      // not permit a copy. After Present, assert actual readiness and retained
      // provider ownership independently instead of interpreting the ticket.
    }
    void wait_batch_completion() {
      checked(batch_queue->Signal(completion.p,++fence_value),"Fence actual large native batch");
      checked(completion->SetEventOnCompletion(fence_value,completion_event),"Observe large batch completion");
      require(WaitForSingleObject(completion_event,3000)==WAIT_OBJECT_0,"Large native batch exceeded three seconds");
    }
    void submit_large_batch(std::uint64_t native_command) {
      require(batch_queue.p && batch_idle.size()==96 && native_command,"Large batch native objects were not prepared");
      if(idle_only_after_submission) {
        // Establish the real producer first. The following oversized batch has
        // no producer/consumer identity and must not revoke that valid capture.
        submit(false);
        batch_queue->ExecuteCommandLists(static_cast<UINT>(batch_idle.size()),batch_idle.data());
        ++submitted_idle_batches;
      } else {
        com_ptr<ID3D12GraphicsCommandList> producer;
        checked(reinterpret_cast<IUnknown *>(native_command)->QueryInterface(IID_PPV_ARGS(producer.put())),
          "Query actual large-batch producer command list");
        checked(commands->Close(),"Close packed producer for one large native submission");
        auto lists=batch_idle;
        require(batch_producer_index<=lists.size(),"Invalid producer position in large native batch");
        lists.insert(lists.begin()+batch_producer_index,producer.p);
        batch_queue->ExecuteCommandLists(static_cast<UINT>(lists.size()),lists.data());
        ++submitted_large_batches;
      }
      wait_batch_completion();
      begin_commands(); // Allocator is idle; outer step records current color/Present.
    }
    void record_large_barrier_batch(std::uint64_t native_command) {
      require(native_command && barrier_scratch,"Large barrier batch requires actual native recording and UAV scratch");
      com_ptr<ID3D12GraphicsCommandList> producer;
      checked(reinterpret_cast<IUnknown *>(native_command)->QueryInterface(IID_PPV_ARGS(producer.put())),
        "Query actual large-barrier producer command list");
      transition(producer.p,selected->resource.p,selected->state,D3D12_RESOURCE_STATE_COPY_SOURCE);
      std::array<D3D12_RESOURCE_BARRIER,257> barriers{};
      for(unsigned i=0;i<256;++i) {
        barriers[i].Type=D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barriers[i].UAV.pResource=barrier_scratch->resource.p;
      }
      auto &last=barriers.back();
      last.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      last.Transition={selected->resource.p,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
        D3D12_RESOURCE_STATE_COPY_SOURCE,selected->state};
      // One actual native call; the depth-relevant final item must survive the
      // snapshot instead of being truncated at the previous 256-item bound.
      producer->ResourceBarrier(static_cast<UINT>(barriers.size()),barriers.data());
      ++recorded_large_barriers;
    }
    void draw_packed_frame() {
      if(rotate) selected=(++rotation_index&1u) ? first.get() : second.get();
      draw(*decoy);
      if(selected->state!=D3D12_RESOURCE_STATE_DEPTH_WRITE)
        transition(commands.p,selected->resource.p,selected->state,D3D12_RESOURCE_STATE_DEPTH_WRITE);
      center_raw=.015625f+float(sequence%8)*.0009765625f;
      write_packed_pixels();
      const auto established=private_input==private_case::common ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
      transition(commands.p,selected->resource.p,D3D12_RESOURCE_STATE_DEPTH_WRITE,established);
      selected->state=established;
      // Both planes share this provider state. Submit the producer before the
      // fresh evaluation recording, so no same-recording transition warms proof.
      submit(false);
      begin_commands();
      const auto native_command=game_native_command;
      if(large_barriers) record_large_barrier_batch(native_command);
      const std::uint32_t encoded=private_input==private_case::common ? chi_present :
        chi_shader_read|(private_input==private_case::shader_read_present ? chi_present : 0);
      if(private_input==private_case::missing)
        checked(selected->resource->SetPrivateData(streamline_v1_state_guid,0,nullptr),"Remove packed source provider state");
      else
        checked(selected->resource->SetPrivateData(streamline_v1_state_guid,sizeof(encoded),&encoded),"Write packed source provider state");
      ticket=0;
      if(capture_v1 && emit) {
        require(native_command,"Packed V1 evaluation lost its native command identity");
        ++v1_capture_calls;
        const auto callback=defer_evaluation_finish ? begin_capture_v1 : capture_v1;
        require(callback,"Deferred V1 capture adapter is missing");
        ticket=callback(native_command,native(*selected),0,&camera,++sequence,valid_evaluation);
        if(defer_evaluation_finish && ticket) {
          // Model a middleware call whose supplied recording is really executed
          // before slEvaluate returns. No completion, readiness or fence is
          // injected: submit closes the producer, executes it and waits for GPU.
          require(finish_capture,"Deferred V1 completion adapter is missing");
          submit(false);
          ++submitted_before_finish;
          finish_capture(ticket,valid_evaluation);
          if(replay_evaluation_recording) {
            // This is a second real execution of the unchanged closed list,
            // after its first execution completed. Keep the stale-replay guard
            // distinct from merely receiving evaluation success later.
            ID3D12CommandList *lists[]{commands.p};
            queue->ExecuteCommandLists(1,lists);
            checked(queue->Signal(completion.p,++fence_value),"Fence replayed native producer");
            checked(completion->SetEventOnCompletion(fence_value,completion_event),"Observe replay completion");
            require(WaitForSingleObject(completion_event,3000)==WAIT_OBJECT_0,"Replayed native producer exceeded three seconds");
            ++replayed_recordings;
          }
          // The outer step now records current color and Present on a fresh
          // list. Reset must retire CPU identity without erasing submitted proof.
          begin_commands();
        }
        else if(large_submission && ticket) submit_large_batch(native_command);
      }
    }
    void draw_frame() {
      if(packed_state_mode) { draw_packed_frame(); return; }
      if(private_state_mode) { draw_private_frame(); return; }
      // Reset event names the real game CL before the native observer sees its
      // subsequent Close/Execute. The test never submits a fabricated command ID.
      const auto native_command = game_native_command;
      if (rotate) selected = (++rotation_index&1u) ? first.get() : second.get();
      draw(*decoy);
      write_depth_pixels();
      // V1's zero state is an omitted state, not permission to assume COMMON.
      // Only actual nonzero transitions in this recording provide that proof.
      if (mode==capture_mode::v1_observed_state) {
        transition(commands.p,selected->resource.p,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
        transition(commands.p,selected->resource.p,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      } else if (mode==capture_mode::v1_common_state) {
        transition(commands.p,selected->resource.p,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COMMON);
      }
      ticket = 0;
      if (capture && emit) {
        require(native_command, "Game command-list native identity was not observed");
        const auto callback=mode==capture_mode::version_two ? capture : capture_v1;
        require(callback,"Version-one immutable capture adapter is missing");
        if (mode==capture_mode::version_two) ++v2_capture_calls;
        else ++v1_capture_calls;
        ticket = callback(native_command,native(*selected),mode==capture_mode::version_two ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : 0,
          &camera,++sequence,valid_evaluation);
      }
      if (mode==capture_mode::v1_common_state)
        transition(commands.p,selected->resource.p,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    bool current() const {
      return captured_ready && captured_render==observed.renders && captured.projection.supplied &&
        captured.source_resource.handle==native(*selected) && captured.width==selected->width && captured.height==selected->height &&
        captured.active_width==selected->width && captured.active_height==selected->height && !captured.x && !captured.y;
    }
    void require_held_mono(const char *message) {
      require(!current() && !captured_ready && !ready(),message);
      if(query_provider_status) {
        sunshine_streamline::provider::source_status status;
        require(query_provider_status(observed.runtime,&status) && status.selected && !status.ready &&
          status.provider==sunshine_scene_depth::provider_kind::streamline && !status.current.resource &&
          !status.current.capture && !status.current.sequence,
          "An unavailable API copy relinquished SL ownership or exposed a historical source as current");
      }
    }
    void tick(const char *phase) {
      step();
      trace<<phase<<','<<GetTickCount64()<<','<<observed.renders<<','<<sequence<<','<<ticket<<','<<unsigned(mode)<<','<<native(*selected)<<','
        <<captured.source_resource.handle<<','<<captured.projection.supplied<<','<<captured_ready<<','<<ready()<<','
        <<scalar("Sunshine_CameraDepthScale")<<','<<zero()[1]<<','<<scalar("Sunshine_CameraStrengthBlend")<<'\n';
      require(trace.good(),"Cannot record direct Streamline trajectory");
    }
    void settle(const char *phase,unsigned timeout=15000) {
      const auto started=GetTickCount64(); unsigned continuous=0;
      do {
        tick(phase);
        continuous=current() && ready() && scalar("Sunshine_CameraStrengthBlend")==1.f ? continuous+1 : 0;
      } while (continuous<6 && GetTickCount64()-started<timeout);
      std::printf("MEASURE %s elapsed_ms=%llu current=%d camera_ready=%d ticket=%llu K=%.9g q0=%.9g\n",phase,
        static_cast<unsigned long long>(GetTickCount64()-started),current(),ready(),static_cast<unsigned long long>(ticket),
        scalar("Sunshine_CameraDepthScale"),zero()[1]);
      require(continuous>=6,"Direct native source did not reach sustained projection rendering");
    }
    void check_projection() {
      require(current() && ready(),"Projection assertion requires actual current native source");
      int basis=-1; float projection[2]{};
      observed.runtime->get_uniform_value_int(uniform("Sunshine_CameraCoordinateBasis"),&basis,1);
      observed.runtime->get_uniform_value_float(uniform("Sunshine_CameraProjection"),projection,2);
      const float reference=scalar("Sunshine_CameraDepthScale");
      require(basis==0 && projection[0]==camera.projection.m[2][2] && projection[1]==1.f/camera.projection.m[3][2] &&
        std::isfinite(reference) && reference>0.f && zero()[0]==.05f && std::abs(reference*zero()[1]-1.f)<2e-6f,
        "Actual shader did not receive the exact current projection and a valid depth reference");
      if(std::isfinite(expected_reference))
        require(std::abs(reference-expected_reference)<=expected_reference*1e-6f,
          "Ordinary rendering replaced the established inverse-depth reference");
      require(selected_binding().handle==captured.shader_resource.handle,"Actual shader binding differs from direct capture packet");
    }
    float prepared_center() {
      const float q=(center_raw-camera.projection.m[2][2])/camera.projection.m[3][2];
      const float reference=std::isfinite(expected_reference) ? expected_reference : scalar("Sunshine_CameraDepthScale");
      return 1.f/(1.f+reference*q);
    }
    void establish_new_reference(const char *phase,float expected) {
      const auto started=GetTickCount64();
      require(recenter(observed.runtime),"Explicit projection Reset reference action was rejected");
      bool saw_unready=false;unsigned continuous=0;
      do {
        tick(phase);
        const auto elapsed=GetTickCount64()-started;
        saw_unready|=!ready();
        require(elapsed>=750 || !ready(),"Reset reference reused old samples instead of gathering four fresh spaced captures");
        continuous=current() && ready() && scalar("Sunshine_CameraStrengthBlend")==1.f ? continuous+1 : 0;
      } while(continuous<6 && GetTickCount64()-started<15000);
      require(saw_unready && continuous>=6,"Explicit Reset reference failed to reinitialize from fresh rendered samples");
      expected_reference=expected;
      check_projection();
      require(std::abs(zero()[1]-1.f/expected)<.001f,"Explicit Reset reference retained the old zero plane");
      std::printf("MEASURE %s elapsed_ms=%llu reference=%.9g q0=%.9g\n",phase,
        static_cast<unsigned long long>(GetTickCount64()-started),scalar("Sunshine_CameraDepthScale"),zero()[1]);
    }
    void verify_depth() {
      check_projection();
      auto *resource=reinterpret_cast<ID3D12Resource *>(captured.resource.handle);
      require(resource && resource->GetDesc().Width==selected->width && resource->GetDesc().Height==selected->height,
        "Direct preserved texture has incorrect geometry");
      const auto bytes=read(resource,D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
      for (unsigned y=0;y<18;++y) for (unsigned x=0;x<32;++x) {
        const unsigned px=(2*x+1)*selected->width/64,py=(2*y+1)*selected->height/36;
        float actual{}; std::memcpy(&actual,bytes.data()+(size_t(py)*selected->width+px)*4,4);
        const bool center=x>=14 && x<18 && y>=7 && y<11;
        const unsigned band=(selected->pattern ? 31-x : x)/8;
        const float expected=center ? center_raw : selected->pattern==2 ? .015625f : .0078125f*(band+1);
        require(std::isfinite(actual) && actual==expected,"Native tagged UAV copy has stale, cleared or wrong spatial depth");
      }
      const auto prepared=read(linear_depth.p);
      const auto desc=linear_depth->GetDesc();
      std::uint16_t half{};
      std::memcpy(&half,prepared.data()+(size_t(desc.Height/2)*desc.Width+desc.Width/2)*4+2,2);
      require(std::abs(half_float(half)-prepared_center())<.002f,
        "Actual shader prepared depth does not use its captured projection coefficients");
    }
    void matrix_change_cases() {
      // Change real camera metadata; never write camera scale/readiness uniforms.
      // The finite far plane changes A as well as B, so retaining or smoothing
      // old conversion coefficients cannot accidentally satisfy this fixture.
      const auto change_camera=[&](float near_distance,float far_distance,float fov) {
        camera.projection={};camera.inverse_projection={};
        camera.near_plane=near_distance;camera.far_plane=far_distance;camera.fov=fov;
        const float sy=1.f/std::tan(fov*.5f),sx=sy/camera.aspect;
        const float A=std::isfinite(far_distance) ? -near_distance/(far_distance-near_distance) : 0.f;
        const float B=std::isfinite(far_distance) ? near_distance*far_distance/(far_distance-near_distance) : near_distance;
        camera.projection.m[0][0]=sx;camera.projection.m[1][1]=sy;
        camera.projection.m[2][2]=A;camera.projection.m[2][3]=1;
        camera.projection.m[3][2]=B;
        camera.inverse_projection.m[0][0]=1.f/sx;camera.inverse_projection.m[1][1]=1.f/sy;
        camera.inverse_projection.m[2][3]=1.f/B;camera.inverse_projection.m[3][2]=1;
        camera.inverse_projection.m[3][3]=-A/B;
      };
      const auto epoch=captured.projection.epoch;
      const auto viewport=captured.projection.viewport;
      const auto check_current_matrix=[&] {
        require(current() && ready() && captured.projection.epoch==epoch && captured.projection.viewport==viewport,
          "Changing camera matrix or rotating depth changed the logical projection domain");
        int basis=-1;float projection[2]{};
        observed.runtime->get_uniform_value_int(uniform("Sunshine_CameraCoordinateBasis"),&basis,1);
        observed.runtime->get_uniform_value_float(uniform("Sunshine_CameraProjection"),projection,2);
        require(basis==0 && projection[0]==camera.projection.m[2][2] && projection[1]==1.f/camera.projection.m[3][2],
          "Actual shader interpolated or retained old A/B instead of using the current camera matrix");
        require(selected_binding().handle==captured.shader_resource.handle,
          "Matrix-transition shader samples a different source than the current API capture");
        require(std::abs(scalar("Sunshine_CameraDepthScale")*zero()[1]-1.f)<2e-6f,
          "Changing current A/B broke zero-plane normalization");
      };
      const auto check_prepared=[&] {
        check_current_matrix();
        const auto bytes=read(linear_depth.p);const auto desc=linear_depth->GetDesc();
        const float A=camera.projection.m[2][2],inverseB=1.f/camera.projection.m[3][2];
        const float reference=1.f/zero()[1];
        for(unsigned y=0;y<18;++y) for(unsigned x=0;x<32;++x) {
          const unsigned px=(2*x+1)*unsigned(desc.Width)/64,py=(2*y+1)*desc.Height/36;
          std::uint16_t half{};std::memcpy(&half,bytes.data()+(size_t(py)*desc.Width+px)*4+2,2);
          const bool center=x>=14 && x<18 && y>=7 && y<11;
          const float raw=center ? center_raw : .0078125f*float((selected->pattern ? 31-x : x)/8+1);
          const float expected=1.f/(1.f+reference*(raw-A)*inverseB);
          require(std::abs(half_float(half)-expected)<.002f,
            "Actual prepared depth does not combine current source pixels/current A/B with the established reference");
        }
      };
      const auto check_hdr=[&] {
        const auto bytes=read(exported.p);float eye_difference=0;
        for(unsigned y=0;y<height;++y) for(unsigned x=0;x<width;++x) for(unsigned c=0;c<4;++c) {
          const float l=channel(bytes,x,y,c),r=channel(bytes,width+x,y,c);
          require(std::isfinite(l) && std::isfinite(r),"Matrix transition produced nonfinite actual HDR stereo");
          eye_difference=std::max(eye_difference,std::abs(l-r));
        }
        require(eye_difference>.01f,"Matrix-transition HDR output never used its actual stereo depth");
      };

      rotate=true;settle("matrix-resource-warmup");check_projection();
      for(unsigned i=0;i<4;++i) {tick("matrix-initial-rotation");check_projection();}
      check_prepared();
      // Keep the first changed matrix adjacent to a valid current tick; slow
      // readback/inspection deliberately does not count as continuous exposure.
      tick("matrix-before-change");Sleep(16);
      expected_reference=std::numeric_limits<float>::quiet_NaN();
      change_camera(.125f,64.f,1.1f);tick("matrix-near-doubled-first-frame");check_current_matrix();
      const float changed_center=(center_raw-camera.projection.m[2][2])/camera.projection.m[3][2];
      require(changed_center<.2f,
        "Matrix fixture did not change decoded distance independently of the reference");
      check_prepared();check_hdr();

      const float held_gain=scalar("Sunshine_CameraDepthScale");
      emit=false;const auto missing_until=GetTickCount64()+700;
      do {
        tick("matrix-transition-provider-gap");
        require(!captured_ready && !ready() && scalar("Sunshine_CameraDepthScale")==held_gain,
          "A provider gap changed the retained projection scale or exposed stale/Generic depth");
      } while(GetTickCount64()<missing_until);
      check_current_mono();
      emit=true;tick("matrix-transition-gap-return");check_current_matrix();
      require(scalar("Sunshine_CameraDepthScale")==held_gain,
        "Returning from a provider gap credited missing time or reset the retained projection gain");

      unsigned seen=0,transition_frames=0;
      const auto deadline=GetTickCount64()+10000;
      do {
        tick("matrix-zero-plane-normalization");check_current_matrix();
        seen|=selected==first.get()?1u:2u;++transition_frames;
      } while((std::abs(zero()[1]-changed_center)>.001f || scalar("Sunshine_CameraStrengthBlend")!=1.f || transition_frames<6) && GetTickCount64()<deadline);
      require(std::abs(zero()[1]-changed_center)<=.001f && seen==3,
        "Independent zero plane did not follow current decoded depth across both real sources");
      check_prepared();check_hdr();

      const float before_fov_zero=zero()[1];
      change_camera(.125f,64.f,.9f);
      for(unsigned i=0;i<6;++i) {
        tick("matrix-FOV-only-change");check_current_matrix();
        require(std::abs(zero()[1]-before_fov_zero)<.001f,
          "Changing only FOV reset the tracked screen plane");
      }
      check_prepared();
      set_float("Depth_Adjustment",0);for(unsigned i=0;i<3;++i)tick("matrix-zero-strength-HDR");
      check_current_mono();
      std::printf("PASS actual matrix transition: reference=%.9g equals reciprocal zero, exact current A/B, current prepared pixels/HDR, zero=%.9g, gaps/rotation/FOV continuity\n",scalar("Sunshine_CameraDepthScale"),zero()[1]);
    }
    std::vector<std::uint8_t> read_packed_plane(ID3D12Resource *texture,unsigned plane,D3D12_RESOURCE_STATES state) {
      const auto desc=texture->GetDesc();
      require(desc.Format==DXGI_FORMAT_R32G8X24_TYPELESS && desc.MipLevels==1 && desc.DepthOrArraySize==1 && plane<2,
        "Packed readback requires the actual one-mip format19 source or capture");
      D3D12_FEATURE_DATA_FORMAT_INFO info{desc.Format,0};
      checked(game->CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO,&info,sizeof(info)),"Query packed source plane count");
      require(info.PlaneCount==2,"D32S8 resource did not expose independent depth and stencil planes");
      D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
      UINT rows{}; UINT64 row_bytes{},total{};
      game->GetCopyableFootprints(&desc,plane,1,0,&footprint,&rows,&row_bytes,&total);
      if(!packed_layout_logged[plane]) {
        std::printf("MEASURE packed readback plane=%u format=%u width=%u height=%u rows=%u row_bytes=%llu row_pitch=%u offset=%llu total=%llu\n",
          plane,unsigned(footprint.Footprint.Format),footprint.Footprint.Width,footprint.Footprint.Height,rows,
          static_cast<unsigned long long>(row_bytes),footprint.Footprint.RowPitch,
          static_cast<unsigned long long>(footprint.Offset),static_cast<unsigned long long>(total));
        packed_layout_logged[plane]=true;
      }
      // D3D12 defines packed D32S8 buffer transfers as an R32 depth plane and
      // an R8 stencil plane, not as interleaved 64-bit pixels. Check the runtime's
      // actual returned layout before interpreting either plane.
      const auto format=footprint.Footprint.Format;
      const bool plane_format=plane ?
        (format==DXGI_FORMAT_R8_TYPELESS || format==DXGI_FORMAT_R8_UINT || format==DXGI_FORMAT_R8_UNORM) :
        (format==DXGI_FORMAT_R32_TYPELESS || format==DXGI_FORMAT_R32_FLOAT || format==DXGI_FORMAT_R32_UINT);
      const auto element_bytes=plane ? sizeof(std::uint8_t) : sizeof(float);
      require(plane_format && rows==desc.Height && row_bytes==desc.Width*element_bytes &&
        footprint.Footprint.Width==desc.Width && footprint.Footprint.Height==desc.Height && footprint.Footprint.Depth==1 &&
        row_bytes<=footprint.Footprint.RowPitch &&
        footprint.Offset+UINT64(rows-1)*footprint.Footprint.RowPitch+row_bytes<=total,
        "Actual packed-plane footprint differs from the documented planar transfer layout");
      com_ptr<ID3D12Resource> staging;
      buffer(staging,total,D3D12_HEAP_TYPE_READBACK);
      begin_commands();
      D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      barrier.Transition={texture,plane,state,D3D12_RESOURCE_STATE_COPY_SOURCE};
      commands->ResourceBarrier(1,&barrier);
      D3D12_TEXTURE_COPY_LOCATION from{},to{};
      from.pResource=texture; from.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; from.SubresourceIndex=plane;
      to.pResource=staging.p; to.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; to.PlacedFootprint=footprint;
      commands->CopyTextureRegion(&to,0,0,0,&from,nullptr);
      std::swap(barrier.Transition.StateBefore,barrier.Transition.StateAfter);
      commands->ResourceBarrier(1,&barrier);
      submit();
      void *mapped{};
      const D3D12_RANGE read_range{static_cast<SIZE_T>(footprint.Offset),static_cast<SIZE_T>(total)};
      checked(staging->Map(0,&read_range,&mapped),"Map completed packed-plane readback");
      std::vector<std::uint8_t> bytes(static_cast<size_t>(row_bytes)*rows);
      for(UINT y=0;y<rows;++y)
        std::memcpy(bytes.data()+size_t(y)*row_bytes,
          static_cast<const std::uint8_t *>(mapped)+footprint.Offset+size_t(y)*footprint.Footprint.RowPitch,static_cast<size_t>(row_bytes));
      const D3D12_RANGE no_write{0,0}; staging->Unmap(0,&no_write);
      return bytes;
    }
    void verify_packed_depth() {
      check_projection();
      auto *resource=reinterpret_cast<ID3D12Resource *>(captured.resource.handle);
      require(resource && resource->GetDesc().Width==selected->width && resource->GetDesc().Height==selected->height,
        "Packed direct capture has incorrect geometry");
      const auto pixels=read_packed_plane(resource,0,D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
      const auto source=read_packed_plane(selected->resource.p,0,selected->state);
      const auto stencil=read_packed_plane(selected->resource.p,1,selected->state);
      for(unsigned y=0;y<18;++y) for(unsigned x=0;x<32;++x) {
        const unsigned px=(2*x+1)*selected->width/64,py=(2*y+1)*selected->height/36;
        const size_t offset=(size_t(py)*selected->width+px)*sizeof(float);
        float actual{},original{};
        std::memcpy(&actual,pixels.data()+offset,sizeof(actual));
        std::memcpy(&original,source.data()+offset,sizeof(original));
        const bool center=x>=14 && x<18 && y>=7 && y<11;
        const float expected=center ? center_raw : .0078125f*float((selected->pattern ? 31-x : x)/8+1);
        require(std::isfinite(actual) && actual==expected && original==expected,
          "Packed capture or restored original depth plane has stale or incorrect pixels");
      }
      for(unsigned y=0;y<selected->height;++y) for(unsigned x=0;x<selected->width;++x) {
        unsigned band=0;
        while(band<3 && x>=(band+1)*selected->width/4) ++band;
        require(stencil[size_t(y)*selected->width+x]==static_cast<std::uint8_t>(selected->stencil_seed+band),
          "Native depth-plane capture changed the original stencil plane");
      }
      const auto prepared=read(linear_depth.p); const auto desc=linear_depth->GetDesc();
      std::uint16_t half{};
      std::memcpy(&half,prepared.data()+(size_t(desc.Height/2)*desc.Width+desc.Width/2)*4+2,sizeof(half));
      require(std::abs(half_float(half)-prepared_center())<.002f,
        "Actual shader did not prepare the current packed depth plane with the supplied projection");
    }
    // Constant-depth rows outside the center patch let the actual source image
    // measure disparity without reproducing the production warp implementation.
    std::array<float,2> shifts(const std::vector<std::uint8_t> &flat,const std::vector<std::uint8_t> &stereo,const char *label) {
      std::array<float,2> result{};
      const int limit=int(width/16), margin=int(width/8);
      for (unsigned eye=0;eye<2;++eye) {
        auto error=[&](float shift) {
          double squared=0,signal=0;
          for (unsigned y : {height/4,height*3/4}) for (int x=margin;x<int(width)-margin;++x) {
            const float sx=x+shift; const auto ix=unsigned(std::floor(sx)); const float t=sx-ix;
            const float a=channel(flat,ix,y,0),b=channel(flat,ix+1,y,0),actual=channel(stereo,eye*width+unsigned(x),y,0);
            require(std::isfinite(actual),"Actual stereo shift has nonfinite color");
            squared+=std::pow(actual-(a+(b-a)*t),2); signal+=double(a)*a+double(b)*b;
          }
          return squared/signal;
        };
        double best=INFINITY; float best_shift=0;
        for (int x=-limit;x<=limit;++x) { const auto e=error(float(x)); if(e<best) { best=e;best_shift=float(x); } }
        const float coarse=best_shift;
        for (int x=-8;x<=8;++x) { const float s=coarse+x*.125f;const auto e=error(s);if(e<best){best=e;best_shift=s;} }
        std::printf("MEASURE direct projection %s eye=%u shift=%.3f error=%.9g\n",label,eye,best_shift,best);
        require(best<.02 && std::abs(best_shift)<limit-1,"Actual constant-plane stereo shift could not be measured");
        result[eye]=best_shift;
      }
      return result;
    }
    void cold_v1_rotation() {
      require(mode==capture_mode::v1_observed_state && rotate && !v1_capture_calls && !v2_capture_calls,
        "Cold V1 regression was warmed by an earlier capture");
      const auto started=GetTickCount64();
      unsigned tagged=0, submitted=0, continuous=0;
      do {
        tick("cold-v1-pretag-rotation");
        const auto source=selected==first.get()?1u:2u;
        tagged|=source;
        if(current()) submitted|=source;
        continuous=current() && ready() && scalar("Sunshine_CameraStrengthBlend")==1.f ? continuous+1 : 0;
      } while(continuous<8 && GetTickCount64()-started<15000);
      std::printf("MEASURE cold V1 elapsed_ms=%llu v1_calls=%u v2_calls=%u tagged_mask=%u copied_mask=%u continuous=%u current=%d camera_ready=%d\n",
        static_cast<unsigned long long>(GetTickCount64()-started),v1_capture_calls,v2_capture_calls,
        tagged,submitted,continuous,current(),ready());
      require(!v2_capture_calls && tagged==3,"Cold V1 regression did not independently exercise both sources");
      require(submitted==3 && continuous>=8,
        "Cold V1 pre-tag rotation never established sustained state proof for both resources");
      for(unsigned i=0;i<8;++i) {
        // Change actual pixels each frame as well as alternating opposite spatial
        // patterns, so stale copies cannot satisfy resource/geometry assertions.
        center_raw=.015625f+float(i+1)*.0009765625f;
        tick("cold-v1-current-pixels");
        require(ticket && current() && ready(),"Cold V1 rotation lost the current native source");
        verify_depth();
      }
      require(!v2_capture_calls,"Cold V1 regression accidentally invoked the V2 capture path");
      std::puts("PASS cold V1-only A/B rotation: transitions precede every tag, both native resources preserve current pixels, zero V2 warming calls");
    }
    void private_state_cases() {
      require(private_state_mode && rotate && !v1_capture_calls && !v2_capture_calls,
        "Provider-private-state regression was warmed by an earlier capture");
      struct private_phase { private_case value; const char *name; };
      for(const auto &phase : {
          private_phase{private_case::shader_read,"private-state-shader-read"},
          private_phase{private_case::shader_read_present,"private-state-shader-read-present-marker"},
          private_phase{private_case::common,"private-state-common-present-marker"}}) {
        private_input=phase.value;
        settle(phase.name);
        unsigned seen=0;
        for(unsigned i=0;i<4;++i) {
          tick(phase.name);
          require(ticket && current() && ready(),"Provider-private state did not preserve the current native depth");
          seen|=selected==first.get()?1u:2u;
          verify_depth();
        }
        require(seen==3,"Provider-private state did not cover both real rotating sources");
      }
      std::puts("PASS provider-private chi PS|NPS with/without Present and Present-only COMMON drive correct native pixels from an earlier submitted recording");
      for(const auto &phase : {
          private_phase{private_case::missing,"private-state-missing"},
          private_phase{private_case::short_value,"private-state-malformed-size"},
          private_phase{private_case::unsupported,"private-state-unsupported-bits"},
          private_phase{private_case::conflicting,"private-state-conflicting-transition"},
          private_phase{private_case::split,"private-state-split-transition"}}) {
        private_input=phase.value;
        for(unsigned i=0;i<4;++i) {
          tick(phase.name);
          require_held_mono(
            "Invalid provider-private state reused stale pixels or selected the generic depth fallback");
        }
        check_current_mono();
      }
      private_input=private_case::shader_read_present;
      settle("private-state-recovery");verify_depth();
      require(!v2_capture_calls && v1_capture_calls,"Provider-private regression invoked a V2 warming path");
      std::puts("PASS missing/malformed/unsupported metadata, conflicting native state and incomplete split barriers remain current-color mono; completed barriers recover with valid provider state");
    }
    void packed_state_cases() {
      require(packed_state_mode && rotate && !v1_capture_calls && !v2_capture_calls,
        "Packed V1 regression was warmed by an earlier capture");
      for(const auto input : {private_case::shader_read_present,private_case::common}) {
        private_input=input;
        const auto phase=input==private_case::common ? "packed-v1-COMMON" : "packed-v1-shader-read";
        settle(phase);
        unsigned seen=0;
        for(unsigned i=0;i<4;++i) {
          tick(phase);
          require(ticket && current() && ready(),"Packed V1 source rotation lost its current independent capture");
          seen|=selected==first.get()?1u:2u;
          verify_packed_depth();
        }
        require(seen==3,"Packed V1 regression did not validate both real format19 sources");
      }
      std::puts("PASS packed R32G8X24/D32S8 private-state rotation: submitted prior-CL state, exact depth-plane pixels, source depth/stencil preservation and actual projection preparation");
      private_input=private_case::missing;
      for(unsigned i=0;i<4;++i) {
        tick("packed-v1-missing-metadata");
        require_held_mono("Packed V1 missing metadata reused stale capture or generic depth");
      }
      check_current_mono();
      private_input=private_case::shader_read_present;
      settle("packed-v1-metadata-recovery"); verify_packed_depth();
      valid_evaluation=false;
      for(unsigned i=0;i<4;++i) {
        tick("packed-v1-failed-evaluation");
        require_held_mono("Failed packed V1 frame reused capture or generic depth");
      }
      check_current_mono();
      valid_evaluation=true;
      settle("packed-v1-evaluation-recovery"); verify_packed_depth();
      require(!v2_capture_calls && v1_capture_calls,"Packed V1 regression invoked a V2 warming path");
      std::puts("PASS packed V1 missing metadata and failed evaluations remain current-color mono; fresh metadata/evaluation restores both rotating sources without V2 warming");
    }
    struct retained_commands {
      // Destruction releases the list before the allocator; these empty or
      // balanced recordings are never submitted and have no pending GPU work.
      com_ptr<ID3D12CommandAllocator> allocator;
      com_ptr<ID3D12GraphicsCommandList> commands;
      std::uint64_t native{};
    };
    using command_pressure = std::vector<std::unique_ptr<retained_commands>>;
    command_pressure retain_commands(unsigned count,bool source_barriers) {
      require(game_native_command && selected,"Command pressure requires the actual warmed game command list/source");
      const auto expected_vtable=*reinterpret_cast<void *const *>(game_native_command);
      com_ptr<ID3D12Device> native_device;
      checked(reinterpret_cast<ID3D12GraphicsCommandList *>(game_native_command)->GetDevice(IID_PPV_ARGS(native_device.put())),
        "Get native device for internal command-list lifetime regression");
      command_pressure result;
      result.reserve(count);
      for(unsigned i=0;i<count;++i) {
        auto entry=std::make_unique<retained_commands>();
        checked(native_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(entry->allocator.put())),
          "Create retained game command allocator");
        checked(native_device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,entry->allocator.p,nullptr,IID_PPV_ARGS(entry->commands.put())),
          "Create retained game command list");
        checked(entry->commands->Close(),"Close new retained command list");
        checked(entry->allocator->Reset(),"Reset never-submitted retained allocator");
        checked(entry->commands->Reset(entry->allocator.p,nullptr),"Reset retained game command list");
        // Bypass ReShade's device wrapper. These objects share the observed
        // native vtable but emit no ReShade creation/destruction callbacks.
        // Only real native Reset/Close hooks maintain their recording state.
        entry->native=reinterpret_cast<std::uint64_t>(entry->commands.p);
        require(entry->native && *reinterpret_cast<void *const *>(entry->native)==expected_vtable &&
          std::none_of(result.begin(),result.end(),[&](const auto &prior){return prior->native==entry->native;}),
          "Retained command lists did not use distinct live objects on the warmed native vtable");
        if(source_barriers) {
          // A known tagged source forces real observer state as well as Close
          // bookkeeping. The balanced pair remains unsubmitted, so its actual
          // GPU state and pixels are unchanged when normal rendering resumes.
          transition(entry->commands.p,selected->resource.p,selected->state,D3D12_RESOURCE_STATE_COPY_SOURCE);
          transition(entry->commands.p,selected->resource.p,D3D12_RESOURCE_STATE_COPY_SOURCE,selected->state);
        }
        checked(entry->commands->Close(),"Close retained command recording");
        result.push_back(std::move(entry));
      }
      std::printf("MEASURE command pressure retained=%zu source_barriers=%u native_vtable=%p submitted=0\n",
        result.size(),unsigned(source_barriers),expected_vtable);
      return result;
    }
    void verify_command_pressure(const char *phase,unsigned retained) {
      // tick resets the normal game list before using its native identity;
      // pressure-list Reset events must never become a synthetic capture owner.
      settle(phase);
      unsigned seen=0;
      for(unsigned i=0;i<12;++i) {
        tick(phase);
        require(ticket && current() && ready() && scalar("Sunshine_CameraStrengthBlend")==1.f,
          "Retained command-list pressure prevented current packed depth from reaching the actual shader");
        seen|=selected==first.get()?1u:2u;
        check_projection();
      }
      require(seen==3,"Command pressure did not retain both current rotating depth sources");
      // Read both spatial/depth patterns, changing center, original stencil and
      // the actual shader's prepared depth while all pressure objects remain live.
      for(unsigned i=0;i<2;++i) {tick(phase);verify_packed_depth();}
      std::printf("PASS %s retained=%u both_sources=1 consecutive_full=12 actual_packed_and_shader_pixels=1\n",phase,retained);
    }
    void command_capacity_cases() {
      require(packed_state_mode && rotate && !v1_capture_calls && !v2_capture_calls,
        "Command capacity regression must begin with the packed V1 path alone");
      private_input=private_case::shader_read_present;
      settle("command-capacity-packed-control");verify_packed_depth();
      std::puts("PASS packed source reaches the actual shader before command pressure");
      constexpr unsigned count=160; // Exceeds the prior shared 128-recording table.
      for(unsigned generation=0;generation<3;++generation) {
        auto retained=retain_commands(count,generation!=0);
        const char *phase=generation==0 ? "command-capacity-live-close" :
          generation==1 ? "command-capacity-live-barriers" : "command-capacity-recreated-barriers";
        verify_command_pressure(phase,static_cast<unsigned>(retained.size()));
        retained.clear();
        std::printf("MEASURE command pressure generation=%u destroyed=%u\n",generation,count);
        verify_command_pressure("command-capacity-after-destroy",0);
      }
      valid_evaluation=false;
      for(unsigned i=0;i<4;++i) {
        tick("command-capacity-failed-current");
        require_held_mono("Command pressure recovery admitted failed or stale depth");
      }
      check_current_mono();
      valid_evaluation=true;
      settle("command-capacity-valid-recovery");verify_packed_depth();
      require(!v2_capture_calls && v1_capture_calls,"Command capacity regression invoked a V2 warming path");
      std::puts("PASS live and destroyed/recreated command-list pressure preserves independent packed current depth, shader projection, and failed-frame mono");
    }
    void pending_evaluation_cases() {
      require(packed_state_mode && rotate && begin_capture_v1 && finish_capture && !v1_capture_calls && !v2_capture_calls,
        "Pending-evaluation regression requires the packed V1 path and split metadata adapters");
      private_input=private_case::shader_read_present;
      settle("pending-evaluation-normal-control");verify_packed_depth();
      std::puts("PASS ordinary packed V1 evaluation reaches the actual shader before submission-order changes");

      defer_evaluation_finish=true;
      settle("pending-evaluation-submitted-before-success");
      unsigned seen=0;
      for(unsigned i=0;i<12;++i) {
        tick("pending-evaluation-submitted-before-success");
        require(ticket && current() && ready() && scalar("Sunshine_CameraStrengthBlend")==1.f,
          "Successful evaluation after actual producer submission did not reach the current shader");
        seen|=selected==first.get()?1u:2u;
        check_projection();
      }
      require(seen==3 && submitted_before_finish>=12,"Pending evaluation did not exercise both real rotating sources after submission");
      for(unsigned i=0;i<2;++i) {tick("pending-evaluation-pixel-check");verify_packed_depth();}
      std::printf("PASS submitted-before-success evaluations=%u both_sources=1 consecutive_full=12 actual_packed_and_shader_pixels=1\n",
        submitted_before_finish);

      valid_evaluation=false;
      const auto before_failed=submitted_before_finish;
      for(unsigned i=0;i<4;++i) {
        tick("pending-evaluation-submitted-before-failure");
        require(ticket,"Submitted failed evaluation lost its nomination ticket");
        require_held_mono("Failed evaluation after submission reused captured or generic depth");
      }
      require(submitted_before_finish==before_failed+4,"Failed evaluation did not follow four actual producer submissions");
      check_current_mono();
      valid_evaluation=true;
      settle("pending-evaluation-failure-recovery");verify_packed_depth();

      replay_evaluation_recording=true;
      for(unsigned i=0;i<4;++i) {
        tick("pending-evaluation-real-producer-replay");
        require(ticket,"Replayed evaluation lost its nomination ticket");
        require_held_mono("Replayed unchanged producer was accepted as a fresh depth frame");
      }
      require(replayed_recordings==4,"Replay regression did not execute four unchanged native recordings twice");
      check_current_mono();
      replay_evaluation_recording=false;
      settle("pending-evaluation-replay-recovery");verify_packed_depth();
      require(!v2_capture_calls && v1_capture_calls,"Pending-evaluation regression invoked a V2 warming path");
      std::printf("PASS submission/completion ordering preserves current packed depth; failed evaluation and %u actual producer replays stay current-color mono; fresh recordings recover\n",
        replayed_recordings);
    }
    void check_provider_status(bool expected_ready) {
      if(!query_provider_status) return;
      sunshine_streamline::provider::source_status status;
      require(query_provider_status(observed.runtime,&status) && status.selected && status.ready==expected_ready,
        "Overlay provider status does not match actual current rendering");
      const auto same=[](const auto &a,const auto &b) {
        return a.resource==b.resource && a.identity==b.identity && a.capture==b.capture && a.sequence==b.sequence &&
          a.width==b.width && a.height==b.height && a.format==b.format && a.tag==b.tag;
      };
      if(expected_ready) {
        require(status.current.resource==native(*selected) && status.current.capture==ticket && status.current.sequence==sequence &&
          status.current.identity && status.current.identity==sunshine_native_identity::resource_cookie(selected->resource.p) &&
          status.current.width==selected->width && status.current.height==selected->height &&
          status.current.format==DXGI_FORMAT_R32G8X24_TYPELESS && status.current.tag==0 &&
          same(status.current,status.last_valid),"Overlay provider status shows the wrong current packed source");
        last_ui_source=status.current;
      } else {
        const sunshine_streamline::provider::source_description empty;
        require(same(status.current,empty) && last_ui_source.resource && same(status.last_valid,last_ui_source),
          "Overlay provider status mislabeled historical depth as current or lost the last valid source");
      }
    }
    void verify_large_batch_phase(const char *phase) {
      settle(phase);
      unsigned seen=0;
      for(unsigned i=0;i<12;++i) {
        tick(phase);
        require(ticket && current() && ready() && scalar("Sunshine_CameraStrengthBlend")==1.f,
          "Large native batch prevented current packed depth from reaching the actual shader");
        seen|=selected==first.get()?1u:2u;
        check_projection();
        check_provider_status(true);
      }
      require(seen==3,"Large native batch did not exercise both rotating depth sources");
      for(unsigned i=0;i<2;++i) {tick(phase);verify_packed_depth();check_provider_status(true);}
      std::printf("PASS %s producer_index=%u producer_batches=%u idle_batches=%u barrier_batches=%u both_sources=1 consecutive_full=12 actual_packed_and_shader_pixels=1\n",
        phase,batch_producer_index,submitted_large_batches,submitted_idle_batches,recorded_large_barriers);
    }
    void large_batch_cases() {
      const char *case_value=std::getenv("SUNSHINE_STREAMLINE_LARGE_BATCH_CASE");
      const bool barriers_only=case_value && std::strcmp(case_value,"barriers")==0;
      require(!case_value || !*case_value || std::strcmp(case_value,"all")==0 || barriers_only,
        "Large batch case must be all or barriers");
      require(packed_state_mode && rotate && !v1_capture_calls && !v2_capture_calls,
        "Large-batch regression must begin with the packed V1 path alone");
      private_input=private_case::shader_read_present;
      settle("large-batch-normal-control");verify_packed_depth();
      std::puts("PASS ordinary packed V1 evaluation reaches the actual shader before large native batches");
      command_pressure retained;
      if(!barriers_only) {
        const auto native_queue=observed.runtime->get_command_queue()->get_native();
        checked(reinterpret_cast<IUnknown *>(native_queue)->QueryInterface(IID_PPV_ARGS(batch_queue.put())),
          "Query actual native queue for large submissions");
        retained=retain_commands(96,false);
        for(const auto &entry:retained) batch_idle.push_back(entry->commands.p);
        large_submission=true;
        const std::array<unsigned,3> positions{0,48,96};
        const std::array<const char *,3> phases{"large-batch-producer-first","large-batch-producer-middle","large-batch-producer-last"};
        for(unsigned i=0;i<positions.size();++i) {
          batch_producer_index=positions[i];
          const auto before=submitted_large_batches;
          verify_large_batch_phase(phases[i]);
          require(submitted_large_batches>=before+14,"Large batch phase skipped real producer submissions");
        }
        idle_only_after_submission=true;
        verify_large_batch_phase("large-batch-idle-after-valid-producer");
        require(submitted_idle_batches>=14,"Large idle-only submission was not exercised");
        idle_only_after_submission=false;
        large_submission=false;
      }
      barrier_scratch=target_uav(32,32,0);
      large_barriers=true;
      verify_large_batch_phase("large-batch-depth-transition-at-barrier-257");
      require(recorded_large_barriers>=14,"Large barrier batch did not execute the final depth transition");

      // Keep the oversized call active while checking rejection of a genuine
      // failed evaluation. A useful generic scene must not replace that frame.
      if(!barriers_only) large_submission=true;
      valid_evaluation=false;
      for(unsigned i=0;i<4;++i) {
        tick("large-batch-failed-evaluation");
        require(ticket,"Large-batch failed evaluation lost its nomination ticket");
        require_held_mono("Large-batch failed evaluation admitted stale or generic depth");
        check_provider_status(false);
      }
      check_current_mono();
      valid_evaluation=true;
      verify_large_batch_phase("large-batch-valid-recovery");
      large_submission=large_barriers=false;
      batch_idle.clear();retained.clear();
      if(query_provider_status) {
        require(manual(observed.runtime,reinterpret_cast<std::uint64_t>(decoy->texture.p)),"Cannot pin generic depth for overlay status check");
        tick("large-batch-ui-manual-pin");
        sunshine_streamline::provider::source_status status;
        require(query_provider_status(observed.runtime,&status) && !status.selected && !status.ready && !status.current.resource,
          "Overlay provider status remained active after explicit generic manual pin");
        require(manual(observed.runtime,0),"Cannot restore API depth after overlay status check");
        settle("large-batch-ui-auto-recovery");verify_packed_depth();check_provider_status(true);
        std::puts("PASS actual overlay snapshot reports current source, historical-only source during failure, and manual override ownership");
      }
      require(!v2_capture_calls && v1_capture_calls,"Large-batch regression invoked a V2 warming path");
      std::printf("PASS large native batches case=%s preserve exact packed/source-stencil/shader pixels and failed-frame mono/recovery\n",
        barriers_only?"barriers":"all");
    }
    void run(bool cold_v1=false,bool private_state=false,bool packed_state=false,bool command_capacity=false,bool pending_evaluation=false,bool large_batch=false,bool matrix_change=false) {
      require(unsigned(cold_v1)+unsigned(private_state)+unsigned(packed_state)+unsigned(command_capacity)+unsigned(pending_evaluation)+unsigned(large_batch)+unsigned(matrix_change)<=1,"Select only one opt-in Streamline state regression");
      if(cold_v1) { mode=capture_mode::v1_observed_state; rotate=true; }
      if(private_state) { private_state_mode=true; mode=capture_mode::v1_unknown_state; rotate=true; }
      if(packed_state || command_capacity || pending_evaluation || large_batch) { packed_state_mode=true; mode=capture_mode::v1_unknown_state; rotate=true; }
      initialize_camera(); create_pipeline(); create_compute();
      const unsigned low_width=width==3840 ? 2228 : width/2,low_height=height==2160 ? 1256 : height/2;
      first=packed_state_mode ? target_packed(low_width,low_height,0) : target_uav(low_width,low_height,0);
      second=packed_state_mode ? target_packed(low_width,low_height,1) : target_uav(low_width,low_height,1); selected=first.get();
      decoy=target(width,height,1,9,false,true);
      trace.open(runtime_directory/"streamline-direct-trajectory.csv");
      trace<<std::setprecision(17)<<"phase,wall_ms,render,evaluation,ticket,capture_mode,tagged,current,projection,capture_ready,camera_ready,K,q0,blend\n";
      reshade::register_event<reshade::addon_event::reset_command_list>(observe_reset);
      render_tracked_depth=[&]{draw_frame();};
      const auto timeout=GetTickCount64()+45000;
      while ((!observed.runtime || !observed.renders) && GetTickCount64()<timeout) step();
      require(observed.runtime && observed.renders && !observed.inject,"Actual Automatic HDR runtime failed to initialize");
      check_unified_addon();
      const auto module=GetModuleHandleW(L"SunshineSBSTest.addon64");
      capture=reinterpret_cast<capture_t>(GetProcAddress(module,"SunshineStreamlineTestCapture"));
      capture_v1=reinterpret_cast<capture_t>(GetProcAddress(module,"SunshineStreamlineTestCaptureV1"));
      query_provider_status=reinterpret_cast<provider_status_t>(GetProcAddress(module,"SunshineDepthTestProviderStatus"));
      require(query_provider_status || !sunshine_camera_fixture::flag("SUNSHINE_STREAMLINE_REQUIRE_PROVIDER_STATUS"),
        "Treatment requires the actual overlay provider status adapter");
      if(large_batch) {
        std::printf("MEASURE overlay provider status adapter=%u; absent adapter skips only UI checks for historical control\n",
          unsigned(query_provider_status!=nullptr));
      }
      if(pending_evaluation) {
        begin_capture_v1=reinterpret_cast<capture_t>(GetProcAddress(module,"SunshineStreamlineTestBeginCaptureV1"));
        finish_capture=reinterpret_cast<finish_capture_t>(GetProcAddress(module,"SunshineStreamlineTestFinishCapture"));
        require(begin_capture_v1 && finish_capture,"Pending-evaluation fixture requires TEST ONLY split capture/completion adapters");
      }
      manual=reinterpret_cast<select_t>(GetProcAddress(module,"SunshineDepthTestSelectManual"));
      recenter=reinterpret_cast<action_t>(GetProcAddress(module,"SunshineGame3DTestRecalibrate"));
      query_frame=reinterpret_cast<frame_t>(GetProcAddress(module,"SunshineDepthTestFrame"));
      require(capture && capture_v1 && manual && recenter && query_frame,"Direct fixture requires TEST ONLY native input and passive frame/UI adapters");
      reshade::register_event<reshade::addon_event::reshade_render_technique>(observe_source);
      set_int("Depth_Map_View",0);set_float("Depth_Adjustment",100);set_float("Sharpen_Power",0);
      find_texture("DoubleTex",exported,width*2,DXGI_FORMAT_R16G16B16A16_FLOAT);
      find_texture("texzBufferN_P",linear_depth,0,DXGI_FORMAT_R16G16_FLOAT);
      if(cold_v1 || private_state || packed_state || command_capacity || pending_evaluation || large_batch) {
        if(cold_v1) cold_v1_rotation();
        else if(private_state) private_state_cases();
        else if(packed_state) packed_state_cases();
        else if(command_capacity) command_capacity_cases();
        else if(pending_evaluation) pending_evaluation_cases();
        else large_batch_cases();
        render_tracked_depth={};
        reshade::unregister_event<reshade::addon_event::reshade_render_technique>(observe_source);
        reshade::unregister_event<reshade::addon_event::reset_command_list>(observe_reset);
        return;
      }
      // Startup center q=.015625/.0625=.25: Zref=1/qref=4.
      expected_reference=4.f;
      // Dead Space startup regression: a device is a valid COM object, but its
      // slot 9 is CreateCommandAllocator, not graphics-command-list Close.
      // A rejected middleware pointer must never install hooks on that vtable.
      require(!capture(reinterpret_cast<std::uint64_t>(game.p),native(*first),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        &camera,++sequence,true),"A device was accepted as a graphics command list");
      for(unsigned i=0;i<4;++i) tick("rejected-device-command-pointer");
      com_ptr<ID3D12CommandAllocator> allocator_after_rejection;
      checked(game->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(allocator_after_rejection.put())),
        "Device allocator creation after rejected middleware command pointer");
      std::puts("PASS actual device rejected as command list; later CreateCommandAllocator keeps its native arguments and succeeds");
      settle("direct-low-UAV-over-native-DSV");verify_depth();
      require(!manual(observed.runtime,native(*first)),"Non-DSV UAV incorrectly appeared in generic depth inventory");
      std::puts("PASS real lower-resolution R32_FLOAT UAV outside generic inventory drives projection stereo over an unrelated full-resolution DSV");
      if(matrix_change) {
        matrix_change_cases();render_tracked_depth={};
        reshade::unregister_event<reshade::addon_event::reshade_render_technique>(observe_source);
        reshade::unregister_event<reshade::addon_event::reset_command_list>(observe_reset);
        return;
      }

      require(manual(observed.runtime,reinterpret_cast<std::uint64_t>(decoy->texture.p)),"Cannot pin the real full-resolution DSV");
      const auto pinned_until=GetTickCount64()+15000;
      do {tick("manual-DSV-over-native-tag");} while((!ready() || !captured_ready || captured.projection.supplied ||
        captured.source_resource.handle!=reinterpret_cast<std::uint64_t>(decoy->texture.p)) && GetTickCount64()<pinned_until);
      require(ready() && captured_ready && !captured.projection.supplied &&
        captured.source_resource.handle==reinterpret_cast<std::uint64_t>(decoy->texture.p),"Native provider ignored an explicit user pin");
      require(manual(observed.runtime,0),"Cannot release explicit DSV pin");
      settle("manual-release-to-native");check_projection();
      std::puts("PASS explicit generic-depth pin takes priority; Auto resumes the direct tagged source and its projection scale");

      rotate=true;
      settle("native-rotation-warmup");
      unsigned seen=0;
      for(unsigned i=0;i<16;++i){tick("native-rotation");check_projection();require(ticket,"Native rotating copy was not recorded");seen|=selected==first.get()?1u:2u;if(i<4)verify_depth();}
      require(seen==3,"Both actual rotating resources were not observed");
      rotate=false;selected=first.get();
      settle("first-source-return");
      const float initial_zero=zero()[1];
      center_raw=.03125f;
      expected_reference=std::numeric_limits<float>::quiet_NaN();
      const auto adapt_until=GetTickCount64()+6000;
      do {tick("center-adaptation");check_projection();} while(GetTickCount64()<adapt_until && std::abs(zero()[1]-.5f)>.001f);
      require(std::abs(zero()[1]-.5f)<.001f && zero()[1]>initial_zero+.1f,"Projection center did not follow a changed scene");
      center_raw=.0234375f;
      establish_new_reference("projection-reset-reference",8.f/3.f);
      verify_depth();
      std::puts("PASS reference follows the tracked zero through scene changes; explicit recenter establishes reference=8/3 from fresh samples");

      // Force a known center on a constant background, then measure actual eye
      // translations at two user strengths using source stripe correlation.
      selected->pattern=2;center_raw=.03125f;
      establish_new_reference("constant-background-reset-reference",2.f);verify_depth();
      set_float("Depth_Adjustment",0);for(unsigned i=0;i<4;++i)tick("strength-zero");
      const auto mono_pixels=check_current_mono();
      set_float("Depth_Adjustment",50);settle("strength-half");check_projection();
      const auto half_pixels=read(exported.p);const auto half_shift=shifts(mono_pixels,half_pixels,"50");
      set_float("Depth_Adjustment",100);settle("strength-full");check_projection();
      const auto full_pixels=read(exported.p);const auto full_shift=shifts(mono_pixels,full_pixels,"100");
      // Background q=.25 and reset center qref=q0=.5 yield Zref=2.
      // The normalized separation is .05*2*(.5-.25)=.025. The per-eye
      // budget is 100 pixels at 2160p, giving 1.25/2.5 pixels at .5/1.
      const float full_expected=.025f*100.f*float(height)/2160.f,half_expected=full_expected*.5f;
      for(unsigned eye=0;eye<2;++eye) {
        const float direction=eye==0 ? 1.f : -1.f;
        require(std::abs(half_shift[eye]-direction*half_expected)<=1.f &&
          std::abs(full_shift[eye]-direction*full_expected)<=1.f,
          "Actual constant-plane disparity differs from the original linear strength range");
        require(std::abs(half_shift[eye])>=.5f && std::abs(full_shift[eye])>std::abs(half_shift[eye])+.5f,
          "Increasing actual user strength did not increase measured stereo disparity");
        require(std::abs(full_shift[eye]-2.f*half_shift[eye])<=1.f,
          "Actual unsaturated constant-plane disparity is not proportional to user strength");
      }
      for(unsigned y=0;y<height;++y)for(unsigned x=0;x<2*width;++x)for(unsigned c=0;c<4;++c)
        require(std::isfinite(channel(full_pixels,x,y,c)),"Projection HDR output has nonfinite RGBA");
      sunshine_parity::write_bytes(runtime_directory/"streamline-direct-final.sbs",full_pixels.data(),full_pixels.size());
      std::puts("PASS actual projection shader preserves zero-strength HDR color and responds proportionally to user strength");

      valid_evaluation=false;
      for(unsigned i=0;i<4;++i) {
        tick("failed-evaluation");
        require_held_mono("Failed API frame selected generic depth or reused stale projection depth");
      }
      check_current_mono();
      valid_evaluation=true;settle("successful-evaluation-recovery");
      emit=false;
      const auto missing_until=GetTickCount64()+1800;
      do {
        tick("missing-provider-frame");
        require_held_mono("Silent active provider fell back to a heuristic buffer");
      } while(GetTickCount64()<missing_until);
      check_current_mono();
      emit=true;settle("missing-provider-frame-recovery");check_projection();
      std::puts("PASS failed and missing active-provider frames remain current-color mono despite a useful generic DSV; recovery preserves projection scale");

      mode=capture_mode::v1_unknown_state;
      for(unsigned i=0;i<4;++i) {
        tick("v1-omitted-state-without-transition");
        require_held_mono("V1 omitted state reused stale proof or silently selected generic depth");
      }
      mode=capture_mode::v1_observed_state;
      settle("v1-observed-nonzero-state");verify_depth();
      require(ticket,"V1 capture did not consume the actual nonzero transition proof");
      mode=capture_mode::v1_common_state;
      for(unsigned i=0;i<4;++i) {
        tick("v1-observed-COMMON-is-not-proof");
        require_held_mono("V1 omitted state treated COMMON as proof or silently selected generic depth");
      }
      mode=capture_mode::v1_observed_state;
      settle("v1-fresh-nonzero-recovery");verify_depth();
      std::puts("PASS V1 omitted state rejects missing/COMMON proof; actual same-recording nonzero transitions capture correct pixels and recover");
      std::puts("PASS actual native copy/queue/fence/projection/HDR regression; no generic DSV dependency or fabricated GPU readiness");
      render_tracked_depth={};
      reshade::unregister_event<reshade::addon_event::reshade_render_technique>(observe_source);
      reshade::unregister_event<reshade::addon_event::reset_command_list>(observe_reset);
    }
  };
}

#ifndef SUNSHINE_DIRECT_RUNTIME_FIXTURE_ONLY
int main(int argc,char **argv) {
  std::setvbuf(stdout,nullptr,_IONBF,0);
  if(argc!=5 && argc!=7) {std::fputs("usage: streamline_direct_runtime <official.dll> <Shaders> <test.addon64> <fresh-output> [width height]\n",stderr);return 2;}
  std::thread([]{Sleep(300000);std::fputs("FAIL direct Streamline runtime watchdog\n",stderr);TerminateProcess(GetCurrentProcess(),124);}).detach();
  try {
    width=argc==7?unsigned(std::stoul(argv[5])):3840;height=argc==7?unsigned(std::stoul(argv[6])):2160;
    require(width>=640 && width<=3840 && height>=360 && height<=2160 && width%4==0 && height%4==0,"Invalid direct fixture dimensions");
    require(sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC") && sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC_ACTIONS_TEST") &&
      !sunshine_camera_fixture::flag("SUNSHINE_DEPTH_BIND_SWITCH_TEST"),"Direct fixture requires Automatic, TEST ONLY actions and mode2 unset");
    require(!fs::exists(fs::absolute(argv[4])),"Fresh isolated runtime output is required");
    direct_fixture fixture;fixture.runtime_directory=fs::absolute(argv[4]);
    fixture.initialize(fs::absolute(argv[1]),fs::absolute(argv[2]),fixture.runtime_directory,2,0,fs::absolute(argv[3]));
    fixture.run(sunshine_camera_fixture::flag("SUNSHINE_STREAMLINE_COLD_V1_TEST"),
      sunshine_camera_fixture::flag("SUNSHINE_STREAMLINE_PRIVATE_STATE_TEST"),
      sunshine_camera_fixture::flag("SUNSHINE_STREAMLINE_PACKED_V1_TEST"),
      sunshine_camera_fixture::flag("SUNSHINE_STREAMLINE_COMMAND_CAPACITY_TEST"),
      sunshine_camera_fixture::flag("SUNSHINE_STREAMLINE_PENDING_EVALUATION_TEST"),
      sunshine_camera_fixture::flag("SUNSHINE_STREAMLINE_LARGE_BATCH_TEST"),
      sunshine_camera_fixture::flag("SUNSHINE_STREAMLINE_MATRIX_CHANGE_TEST"));return 0;
  } catch(const std::exception &error){std::fprintf(stderr,"FAIL %s\n",error.what());return 1;}
}
#endif
