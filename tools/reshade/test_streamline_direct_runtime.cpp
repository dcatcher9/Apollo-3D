// SPDX-License-Identifier: GPL-3.0-only
// Real UAV scene depth -> native middleware copy/submission -> projection
// controller -> native Game 3D HDR rendering with no installed FX. Only
// immutable middleware metadata is synthetic; no depth binding, readiness,
// sampler result or fence is injected.
#include "test_raw_runtime_fixture.h"
#include "test_game3d_native_observation.h"
#include "depth_addon.h"
#include "streamline_depth_provider.h"
#include "native_resource_identity.h"
#include <deque>

namespace {
  using capture_t = std::uint64_t (*)(std::uint64_t, std::uint64_t, std::uint32_t,
    const sunshine_streamline::camera_data *, std::uint64_t, bool);
  using finish_capture_t = void (*)(std::uint64_t, bool);
  using provider_status_t = BOOL (*)(api::effect_runtime *, sunshine_streamline::provider::source_status *);
  using select_t = BOOL (*)(api::effect_runtime *, std::uint64_t);
  using action_t = BOOL (*)(api::effect_runtime *);
  std::uint64_t game_native_command{};
  constexpr GUID streamline_v1_state_guid{0x694b3e1c,0x0e33,0x416f,{0xba,0x83,0xfe,0x24,0x8d,0xa1,0xe8,0x5d}};
  constexpr std::uint32_t chi_shader_read=(1u<<5)|(1u<<6), chi_present=1u<<18;
  void observe_reset(api::command_list *commands) { game_native_command = commands->get_native(); }

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
    // Each Present's native render, observed without FX (only the direct
    // runtime's own run(); derived fixtures keep their own observers).
    native_game3d_observer direct_native{*this};
    sunshine_game3d::test::last_render latest;
    sunshine_depth::frame_depth captured;
    bool captured_ready{};
    std::uint64_t rendered_sequence{};
    // What each recent Present drew, to check the render a Dump 3D consumed.
    struct drawn_frame { std::uint64_t frame{}, source{}; unsigned pattern{}; float center{}; };
    std::deque<drawn_frame> drawn_frames;
    bool ready() const { return latest.rendered && latest.parameters.depth_ready && latest.parameters.camera_ready; }
    float gain() const { return latest.parameters.depth_scale; }
    std::array<float,2> zero() const { return latest.parameters.convergence; }
    float blend() const { return latest.parameters.strength_blend; }
    // docs/reshade-sbs.md, raw-depth automation, on inverse distance
    // q=(raw-A)/B of the current camera: gain L/Q from the full-image maximum
    // Q and the zero at the contrast midpoint b+0.5*sum(d*d)/sum(d), b the
    // minimum and d=q-b. expect_gain holds while Q is unchanged since the last
    // fresh reference; expect_zero once the zero has settled at the current
    // content's midpoint. Capture-only variants vary the center every frame.
    bool expect_gain = true, expect_zero = false;
    float raw_at(const uav_target &target, unsigned x, unsigned y, float center) const {
      const float u = (float(x)+.5f)/float(target.width), v = (float(y)+.5f)/float(target.height);
      const unsigned cx = std::min(31u, unsigned(u*32.f)), cy = std::min(17u, unsigned(v*18.f));
      if (cx >= 14 && cx < 18 && cy >= 7 && cy < 11) return center;
      return target.pattern == 2 ? .015625f : .0078125f*float((target.pattern ? 31-cx : cx)/8+1);
    }
    std::array<double,2> oracle(const uav_target &target, float center) const {
      const double A = camera.projection.m[2][2], inverseB = 1./camera.projection.m[3][2];
      double minimum = INFINITY, maximum = -INFINITY, sum = 0, square = 0;
      for (unsigned y = 0; y < target.height; ++y) for (unsigned x = 0; x < target.width; ++x) {
        const double q = (raw_at(target, x, y, center)-A)*inverseB;
        minimum = std::min(minimum, q); maximum = std::max(maximum, q);
      }
      for (unsigned y = 0; y < target.height; ++y) for (unsigned x = 0; x < target.width; ++x) {
        const double d = (raw_at(target, x, y, center)-A)*inverseB-minimum;
        sum += d; square += d*d;
      }
      const double c = (double(height)/2160.*100.)/double(width);
      const double limit = std::min({std::min(1.5, .04/c), std::min(2.5, .04/c), .01/c})/.05;
      return {limit/maximum, minimum+.5*square/sum};
    }
    // The nearest q of the current draw: the bands' farthest raw or a nearer center.
    double nearest_gain(float center) const {
      const double A = camera.projection.m[2][2], inverseB = 1./camera.projection.m[3][2];
      const double band = selected->pattern == 2 ? .015625 : .03125;
      const double c = (double(height)/2160.*100.)/double(width);
      const double limit = std::min({std::min(1.5, .04/c), std::min(2.5, .04/c), .01/c})/.05;
      return limit/((std::max(band, double(center))-A)*inverseB);
    }
    static bool close(double actual, double expected) {
      return std::isfinite(actual) && std::abs(actual-expected) <= 1e-5*std::max(1., std::abs(expected));
    }
    std::vector<std::uint8_t> check_exported_mono() { return check_current_mono(direct_native.exported()); }

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
      return captured_ready && captured.projection.supplied &&
        captured.source_resource.handle==native(*selected) && captured.width==selected->width && captured.height==selected->height &&
        captured.active_width==selected->width && captured.active_height==selected->height && !captured.x && !captured.y;
    }
    // Failed, missing or refused input holds the newest completed copy,
    // unchanged and still owned by SL, within its source age bound, then
    // renders current-color mono (docs/reshade-sbs.md); never Generic depth or
    // an older copy, and no hold after mono. hold_begin starts each phase;
    // the result counts this Present as mono.
    sunshine_streamline::provider::source_status provider;
    std::uint64_t present_started{}, fresh_ended{}, fresh_sequence{}, fresh_source{};
    bool hold_saw_mono{};
    void hold_begin() { hold_saw_mono=false; }
    bool require_held_mono(const char *message) {
      if(captured_ready || ready()) {
        require(!hold_saw_mono && captured_ready && captured.reused_depth && ready() &&
          captured.source_resource.handle==fresh_source &&
          present_started<fresh_ended+sunshine_scene_depth::maximum_source_age_ms,message);
        if(query_provider_status)
          require(provider.selected && provider.ready && provider.reused_depth &&
            provider.provider==sunshine_scene_depth::provider_kind::streamline && provider.current.sequence==fresh_sequence,
            "A held API copy is not the newest completed SL capture");
        return false;
      }
      hold_saw_mono=true;
      require(!current() && !captured_ready && !ready(),message);
      if(query_provider_status) {
        if(!provider.selected || provider.ready || provider.current.resource || provider.current.capture || provider.current.sequence)
          std::printf("MEASURE unavailable SL status selected=%u ready=%u reused=%u provider=%u resource=0x%llx capture=%llu sequence=%llu\n",
            unsigned(provider.selected),unsigned(provider.ready),unsigned(provider.reused_depth),unsigned(provider.provider),
            static_cast<unsigned long long>(provider.current.resource),static_cast<unsigned long long>(provider.current.capture),
            static_cast<unsigned long long>(provider.current.sequence));
        require(provider.selected && !provider.ready &&
          provider.provider==sunshine_scene_depth::provider_kind::streamline && !provider.current.resource &&
          !provider.current.capture && !provider.current.sequence,
          "An unavailable API copy relinquished SL ownership or exposed a historical source as current");
      }
      return true;
    }
    void tick(const char *phase) {
      present_started=GetTickCount64();
      step(); direct_native.no_effects();
      latest=direct_native.last_render();
      require(latest.sequence>rendered_sequence,"A Present had no native Game 3D render to observe");
      rendered_sequence=latest.sequence;
      captured=latest.depth; captured_ready=captured.ready;
      if(query_provider_status) require(query_provider_status(observed.runtime,&provider),"SL provider UI observation failed");
      if(captured_ready && !captured.reused_depth) {
        fresh_ended=GetTickCount64(); fresh_source=captured.source_resource.handle;
        fresh_sequence=query_provider_status ? provider.current.sequence : 0;
      }
      if(captured_ready) {
        drawn_frames.push_back({captured.frame_index,native(*selected),selected->pattern,center_raw});
        if(drawn_frames.size()>64) drawn_frames.pop_front();
      }
      trace<<phase<<','<<GetTickCount64()<<','<<latest.sequence<<','<<sequence<<','<<ticket<<','<<unsigned(mode)<<','<<native(*selected)<<','
        <<captured.source_resource.handle<<','<<captured.projection.supplied<<','<<captured_ready<<','<<ready()<<','
        <<gain()<<','<<zero()[1]<<','<<blend()<<'\n';
      require(trace.good(),"Cannot record direct Streamline trajectory");
    }
    void settle(const char *phase,unsigned timeout=15000) {
      const auto started=GetTickCount64(); unsigned continuous=0;
      do {
        tick(phase);
        continuous=current() && ready() && blend()==1.f ? continuous+1 : 0;
      } while (continuous<6 && GetTickCount64()-started<timeout);
      std::printf("MEASURE %s elapsed_ms=%llu current=%d camera_ready=%d ticket=%llu K=%.9g q0=%.9g\n",phase,
        static_cast<unsigned long long>(GetTickCount64()-started),current(),ready(),static_cast<unsigned long long>(ticket),
        gain(),zero()[1]);
      require(continuous>=6,"Direct native source did not reach sustained projection rendering");
    }
    void check_projection() {
      require(current() && ready(),"Projection assertion requires actual current native source");
      const auto &p=latest.parameters;
      require(p.coordinate_basis==0 && p.projection[0]==camera.projection.m[2][2] && p.projection[1]==1.f/camera.projection.m[3][2] &&
        std::isfinite(p.depth_scale) && p.depth_scale>0.f && p.convergence[0]==.05f && std::isfinite(p.convergence[1]),
        "The native render did not receive the exact current projection and a valid depth reference");
      if(expect_gain)
        require(close(gain(),nearest_gain(center_raw)),"Ordinary rendering replaced the established gain L/Q of the nearest depth");
      if(expect_zero)
        require(close(zero()[1],oracle(*selected,center_raw)[1]),"The screen plane is not the current depth's contrast midpoint");
    }
    void establish_new_reference(const char *phase) {
      const auto started=GetTickCount64();
      require(recenter(observed.runtime),"Explicit projection Reset reference action was rejected");
      bool saw_unready=false;unsigned continuous=0;
      do {
        tick(phase);
        const auto elapsed=GetTickCount64()-started;
        saw_unready|=!ready();
        require(elapsed>=750 || !ready(),"Reset reference reused old samples instead of gathering four fresh spaced captures");
        continuous=current() && ready() && blend()==1.f ? continuous+1 : 0;
      } while(continuous<6 && GetTickCount64()-started<15000);
      require(saw_unready && continuous>=6,"Explicit Reset reference failed to reinitialize from fresh rendered samples");
      expect_gain=expect_zero=true;
      check_projection();
      std::printf("MEASURE %s elapsed_ms=%llu reference=%.9g q0=%.9g\n",phase,
        static_cast<unsigned long long>(GetTickCount64()-started),gain(),zero()[1]);
    }
    // One production Dump 3D capture: the exact depth and projection one
    // native render consumed, matched to what its own Present drew.
    // Which rotating source (1 first, 2 second) the last capture verified.
    unsigned consumed_source{};
    native_render consumed_depth(const char *phase) {
      const auto consumed=direct_native.capture(phase,false,[&]{tick(phase);});
      const auto drawn=std::find_if(drawn_frames.rbegin(),drawn_frames.rend(),[&](const drawn_frame &value) {
        return value.frame==consumed.frame_index && value.source==consumed.source_resource;
      });
      require(consumed.depth_ready && drawn!=drawn_frames.rend(),"The captured native render consumed no observed current API depth");
      const auto &target=*(drawn->source==native(*first) ? first : second);
      consumed_source=drawn->source==native(*first) ? 1u : 2u;
      require(consumed.width==target.width && consumed.height==target.height && consumed.projection_supplied &&
        consumed.coordinate_basis==0 && consumed.projection[0]==camera.projection.m[2][2] &&
        consumed.projection[1]==1.f/camera.projection.m[3][2],
        "The native render did not consume this capture with its supplied projection coefficients");
      for (unsigned y=0;y<18;++y) for (unsigned x=0;x<32;++x) {
        const unsigned px=(2*x+1)*target.width/64,py=(2*y+1)*target.height/36;
        const float actual=consumed.depth(px,py);
        const bool center=x>=14 && x<18 && y>=7 && y<11;
        const unsigned band=(drawn->pattern ? 31-x : x)/8;
        const float expected=center ? drawn->center : drawn->pattern==2 ? .015625f : .0078125f*(band+1);
        require(std::isfinite(actual) && actual==expected,"Native tagged copy has stale, cleared or wrong spatial depth");
      }
      return consumed;
    }
    unsigned verify_depth() {
      check_projection();
      consumed_depth("direct-consumed-depth");
      return consumed_source;
    }
    // Verifies the consumed pixels of both rotating sources: a capture takes
    // several Presents, so consecutive captures can land on one source.
    void verify_both(const char *phase,bool packed,const std::function<void()> &each={}) {
      unsigned verified=0;
      for(unsigned i=0;verified!=3;++i) {
        require(i<16,"The consumed depth of both rotating sources was never verified");
        tick(phase);
        if(i&1) tick(phase); // A capture spans a fixed number of Presents: shift its parity.
        verified|=packed ? verify_packed_depth() : verify_depth();
        if(each) each();
      }
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
        const auto &p=latest.parameters;
        require(p.coordinate_basis==0 && p.projection[0]==camera.projection.m[2][2] && p.projection[1]==1.f/camera.projection.m[3][2],
          "The native render interpolated or retained old A/B instead of using the current camera matrix");
        require(std::isfinite(p.depth_scale) && p.depth_scale>0.f && p.convergence[0]==.05f && std::isfinite(p.convergence[1]),
          "Changing current A/B left no valid gain or screen plane");
      };
      // The consumed depth and A/B of one render, as its own Present drew them.
      const auto check_prepared=[&] {
        check_current_matrix();
        consumed_depth("matrix-consumed-depth");
      };
      const auto check_hdr=[&] {
        const auto bytes=direct_native.exported();float eye_difference=0;
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
      // A new matrix changes every decoded distance, its nearest Q among them:
      // the gain and zero then adapt toward the new targets.
      expect_gain=expect_zero=false;
      const auto before=oracle(*selected,center_raw);
      change_camera(.125f,64.f,1.1f);tick("matrix-near-doubled-first-frame");check_current_matrix();
      const auto changed=oracle(*selected,center_raw);
      require(std::abs(changed[1]-before[1])>.05 && std::abs(changed[0]-before[0])>.5,
        "Matrix fixture did not change decoded distance independently of the reference");
      check_prepared();check_hdr();
      // The readbacks above present nothing for longer than the one-second
      // association timeout; resume the provider before the timed gap.
      settle("matrix-before-provider-gap");

      const float held_gain=gain();
      emit=false;const auto missing_until=GetTickCount64()+700;
      hold_begin();
      do {
        tick("matrix-transition-provider-gap");
        require_held_mono("A provider gap exposed stale or Generic depth");
        require(direct_native.automatic().scale==held_gain && (!ready() || gain()==held_gain),
          "A provider gap changed the retained projection scale");
      } while(GetTickCount64()<missing_until);
      require(hold_saw_mono,"A 700 ms provider gap outlived the source age bound");
      check_exported_mono();
      emit=true;tick("matrix-transition-gap-return");check_current_matrix();
      require(gain()==held_gain,
        "Returning from a provider gap credited missing time or reset the retained projection gain");

      unsigned seen=0,transition_frames=0;
      const auto deadline=GetTickCount64()+10000;
      do {
        tick("matrix-zero-plane-follows-depth");check_current_matrix();
        seen|=selected==first.get()?1u:2u;++transition_frames;
      } while((!close(zero()[1],changed[1]) || !close(gain(),changed[0]) || blend()!=1.f || transition_frames<6) &&
        GetTickCount64()<deadline);
      require(close(zero()[1],changed[1]) && close(gain(),changed[0]) && seen==3,
        "Gain and zero plane did not follow current decoded depth across both real sources");
      expect_gain=expect_zero=true;
      check_prepared();check_hdr();

      const float before_fov_zero=zero()[1];
      change_camera(.125f,64.f,.9f);
      for(unsigned i=0;i<6;++i) {
        tick("matrix-FOV-only-change");check_current_matrix();
        require(std::abs(zero()[1]-before_fov_zero)<.001f,
          "Changing only FOV reset the tracked screen plane");
      }
      check_prepared();
      direct_native.set_strength(0);for(unsigned i=0;i<3;++i)tick("matrix-zero-strength-HDR");
      check_exported_mono();
      std::printf("PASS actual matrix transition: gain=%.9g and zero=%.9g follow the new decoded depth, exact current A/B, consumed pixels/HDR, gaps/rotation/FOV continuity\n",gain(),zero()[1]);
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
    unsigned verify_packed_depth() {
      check_projection();
      consumed_depth("packed-consumed-depth");
      const unsigned verified=consumed_source;
      const auto source=read_packed_plane(selected->resource.p,0,selected->state);
      const auto stencil=read_packed_plane(selected->resource.p,1,selected->state);
      for(unsigned y=0;y<18;++y) for(unsigned x=0;x<32;++x) {
        const unsigned px=(2*x+1)*selected->width/64,py=(2*y+1)*selected->height/36;
        const size_t offset=(size_t(py)*selected->width+px)*sizeof(float);
        float original{};
        std::memcpy(&original,source.data()+offset,sizeof(original));
        const bool center=x>=14 && x<18 && y>=7 && y<11;
        const float expected=center ? center_raw : .0078125f*float((selected->pattern ? 31-x : x)/8+1);
        require(original==expected,"Restored original packed depth plane has stale or incorrect pixels");
      }
      for(unsigned y=0;y<selected->height;++y) for(unsigned x=0;x<selected->width;++x) {
        unsigned band=0;
        while(band<3 && x>=(band+1)*selected->width/4) ++band;
        require(stencil[size_t(y)*selected->width+x]==static_cast<std::uint8_t>(selected->stencil_seed+band),
          "Native depth-plane capture changed the original stencil plane");
      }
      return verified;
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
        continuous=current() && ready() && blend()==1.f ? continuous+1 : 0;
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
        unsigned seen=0,verified=0;
        for(unsigned i=0;i<4 || verified!=3;++i) {
          require(i<16,"Provider-private state did not validate both real rotating sources");
          tick(phase.name);
          require(ticket && current() && ready(),"Provider-private state did not preserve the current native depth");
          seen|=selected==first.get()?1u:2u;
          if(i&1) tick(phase.name); // A capture spans a fixed number of Presents: shift its parity.
          verified|=verify_depth();
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
        hold_begin();
        for(unsigned mono=0;mono<4;) {
          tick(phase.name);
          mono+=require_held_mono(
            "Invalid provider-private state reused stale pixels or selected the generic depth fallback");
        }
        check_exported_mono();
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
        unsigned seen=0,verified=0;
        for(unsigned i=0;i<4 || verified!=3;++i) {
          require(i<16,"Packed V1 regression did not validate both real format19 sources");
          tick(phase);
          require(ticket && current() && ready(),"Packed V1 source rotation lost its current independent capture");
          seen|=selected==first.get()?1u:2u;
          if(i&1) tick(phase); // A capture spans a fixed number of Presents: shift its parity.
          verified|=verify_packed_depth();
        }
        require(seen==3,"Packed V1 regression did not cover both real format19 sources");
      }
      std::puts("PASS packed R32G8X24/D32S8 private-state rotation: submitted prior-CL state, exact consumed depth-plane pixels and projection, source depth/stencil preservation");
      private_input=private_case::missing;
      hold_begin();
      for(unsigned mono=0;mono<4;) {
        tick("packed-v1-missing-metadata");
        mono+=require_held_mono("Packed V1 missing metadata reused stale capture or generic depth");
      }
      check_exported_mono();
      private_input=private_case::shader_read_present;
      settle("packed-v1-metadata-recovery"); verify_packed_depth();
      valid_evaluation=false;
      hold_begin();
      for(unsigned mono=0;mono<4;) {
        tick("packed-v1-failed-evaluation");
        mono+=require_held_mono("Failed packed V1 frame reused capture or generic depth");
      }
      check_exported_mono();
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
        require(ticket && current() && ready() && blend()==1.f,
          "Retained command-list pressure prevented current packed depth from reaching the actual shader");
        seen|=selected==first.get()?1u:2u;
        check_projection();
      }
      require(seen==3,"Command pressure did not retain both current rotating depth sources");
      // Read both spatial/depth patterns, changing center, original stencil and
      // the depth and projection a native render consumed while all pressure
      // objects remain live.
      verify_both(phase,true);
      std::printf("PASS %s retained=%u both_sources=1 consecutive_full=12 consumed_packed_pixels_and_projection=1\n",phase,retained);
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
      hold_begin();
      for(unsigned mono=0;mono<4;) {
        tick("command-capacity-failed-current");
        mono+=require_held_mono("Command pressure recovery admitted failed or stale depth");
      }
      check_exported_mono();
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
        require(ticket && current() && ready() && blend()==1.f,
          "Successful evaluation after actual producer submission did not reach the current shader");
        seen|=selected==first.get()?1u:2u;
        check_projection();
      }
      require(seen==3 && submitted_before_finish>=12,"Pending evaluation did not exercise both real rotating sources after submission");
      verify_both("pending-evaluation-pixel-check",true);
      std::printf("PASS submitted-before-success evaluations=%u both_sources=1 consecutive_full=12 consumed_packed_pixels_and_projection=1\n",
        submitted_before_finish);

      valid_evaluation=false;
      const auto before_failed=submitted_before_finish;
      unsigned failed_presents=0;
      hold_begin();
      for(unsigned mono=0;mono<4;++failed_presents) {
        tick("pending-evaluation-submitted-before-failure");
        require(ticket,"Submitted failed evaluation lost its nomination ticket");
        mono+=require_held_mono("Failed evaluation after submission reused captured or generic depth");
      }
      require(submitted_before_finish==before_failed+failed_presents,"Failed evaluation did not follow an actual producer submission every Present");
      check_exported_mono();
      valid_evaluation=true;
      settle("pending-evaluation-failure-recovery");verify_packed_depth();

      replay_evaluation_recording=true;
      unsigned replay_presents=0;
      hold_begin();
      for(unsigned mono=0;mono<4;++replay_presents) {
        tick("pending-evaluation-real-producer-replay");
        require(ticket,"Replayed evaluation lost its nomination ticket");
        mono+=require_held_mono("Replayed unchanged producer was accepted as a fresh depth frame");
      }
      require(replayed_recordings==replay_presents,"Replay regression did not execute every unchanged native recording twice");
      check_exported_mono();
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
        require(ticket && current() && ready() && blend()==1.f,
          "Large native batch prevented current packed depth from reaching the actual shader");
        seen|=selected==first.get()?1u:2u;
        check_projection();
        check_provider_status(true);
      }
      require(seen==3,"Large native batch did not exercise both rotating depth sources");
      verify_both(phase,true,[&]{check_provider_status(true);});
      std::printf("PASS %s producer_index=%u producer_batches=%u idle_batches=%u barrier_batches=%u both_sources=1 consecutive_full=12 consumed_packed_pixels_and_projection=1\n",
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
      hold_begin();
      for(unsigned mono=0;mono<4;) {
        tick("large-batch-failed-evaluation");
        require(ticket,"Large-batch failed evaluation lost its nomination ticket");
        const bool unready=require_held_mono("Large-batch failed evaluation admitted stale or generic depth");
        if(unready) check_provider_status(false);
        mono+=unready;
      }
      check_exported_mono();
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
      require(capture && capture_v1 && manual && recenter,"Direct fixture requires TEST ONLY native input and UI adapters");
      // Remove the boot FX and present until the renderer has compiled, all
      // without middleware calls: the cold and private-state regressions
      // require an unwarmed capture.
      emit=false;
      direct_native.start();direct_native.attach_export();direct_native.set_strength(100);
      rendered_sequence=direct_native.await_render([&]{step();});
      emit=true;
      // The capture-only variants vary the center every frame below the
      // bands' nearest depth: the gain stays L/Q while the zero follows.
      if(cold_v1 || private_state || packed_state || command_capacity || pending_evaluation || large_batch) {
        if(cold_v1) cold_v1_rotation();
        else if(private_state) private_state_cases();
        else if(packed_state) packed_state_cases();
        else if(command_capacity) command_capacity_cases();
        else if(pending_evaluation) pending_evaluation_cases();
        else large_batch_cases();
        render_tracked_depth={};
        reshade::unregister_event<reshade::addon_event::reset_command_list>(observe_reset);
        return;
      }
      // Startup: a fresh reference of the static draw sets gain L/Q and the
      // contrast-midpoint zero.
      expect_zero=true;
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
      // A nearer center keeps the nearest Q, hence the gain, and moves the
      // zero to the changed contrast midpoint.
      center_raw=.03125f;
      expect_zero=false;
      const double changed_zero=oracle(*selected,center_raw)[1];
      const auto adapt_until=GetTickCount64()+6000;
      do {tick("center-adaptation");check_projection();} while(GetTickCount64()<adapt_until && !close(zero()[1],changed_zero));
      require(close(zero()[1],changed_zero) && zero()[1]>initial_zero,"Projection screen plane did not follow a changed scene");
      center_raw=.0234375f;
      establish_new_reference("projection-reset-reference");
      verify_depth();
      std::puts("PASS gain holds L/Q while the zero follows a changed scene; explicit recenter establishes the current midpoint from fresh samples");

      // Force a known center on a constant background, then measure actual eye
      // translations at two user strengths using source stripe correlation.
      selected->pattern=2;center_raw=.03125f;
      establish_new_reference("constant-background-reset-reference");verify_depth();
      direct_native.set_strength(0);for(unsigned i=0;i<4;++i)tick("strength-zero");
      const auto mono_pixels=check_exported_mono();
      direct_native.set_strength(50);settle("strength-half");check_projection();
      const auto half_pixels=direct_native.exported();const auto half_shift=shifts(mono_pixels,half_pixels,"50");
      direct_native.set_strength(100);settle("strength-full");check_projection();
      const auto full_pixels=direct_native.exported();const auto full_shift=shifts(mono_pixels,full_pixels,"100");
      // Background q=.25 and the reset reference (gain L/Q, zero q0) give the
      // normalized separation .05*gain*(q0-.25). The per-eye budget is 100
      // pixels at 2160p.
      const auto reference=oracle(*selected,center_raw);
      const float full_expected=float(.05*reference[0]*(reference[1]-.25))*100.f*float(height)/2160.f,half_expected=full_expected*.5f;
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
      hold_begin();
      for(unsigned mono=0;mono<4;) {
        tick("failed-evaluation");
        mono+=require_held_mono("Failed API frame selected generic depth or reused stale projection depth");
      }
      check_exported_mono();
      valid_evaluation=true;settle("successful-evaluation-recovery");
      emit=false;
      const auto silence_began=GetTickCount64();
      hold_begin();
      for(unsigned mono=0;mono<4;) {
        tick("missing-provider-frame");
        mono+=require_held_mono("Silent active provider fell back to a heuristic buffer");
      }
      check_exported_mono();
      // A silent provider keeps the queue mono only for the association
      // timeout (one second without an evaluation); then the fixture's useful
      // Generic scene may take over.
      bool association_released=!query_provider_status;
      do {
        tick("missing-provider-frame");
        require(!current(),"A silent provider exposed its stale capture as current");
        if(!query_provider_status) continue;
        if(GetTickCount64()-silence_began<900)
          require(provider.selected && !provider.ready && !ready(),"Silent provider released its association before the timeout");
        if(!provider.selected) association_released=true;
        else require(!association_released,"A released SL association returned without an evaluation");
      } while(GetTickCount64()-silence_began<1800);
      require(association_released,"Silent SL provider kept the queue mono past the association timeout");
      emit=true;settle("missing-provider-frame-recovery");check_projection();
      std::puts("PASS failed and missing active-provider frames hold only the newest completed copy within its age bound, then render current-color mono despite a useful generic DSV; one second of silence releases the association; recovery preserves projection scale");

      mode=capture_mode::v1_unknown_state;
      hold_begin();
      for(unsigned mono=0;mono<4;) {
        tick("v1-omitted-state-without-transition");
        mono+=require_held_mono("V1 omitted state reused stale proof or silently selected generic depth");
      }
      mode=capture_mode::v1_observed_state;
      settle("v1-observed-nonzero-state");verify_depth();
      require(ticket,"V1 capture did not consume the actual nonzero transition proof");
      mode=capture_mode::v1_common_state;
      hold_begin();
      for(unsigned mono=0;mono<4;) {
        tick("v1-observed-COMMON-is-not-proof");
        mono+=require_held_mono("V1 omitted state treated COMMON as proof or silently selected generic depth");
      }
      mode=capture_mode::v1_observed_state;
      settle("v1-fresh-nonzero-recovery");verify_depth();
      std::puts("PASS V1 omitted state rejects missing/COMMON proof; actual same-recording nonzero transitions capture correct pixels and recover");
      std::puts("PASS actual native copy/queue/fence/projection/HDR regression; no generic DSV dependency or fabricated GPU readiness; no FX");
      render_tracked_depth={};
      reshade::unregister_event<reshade::addon_event::reset_command_list>(observe_reset);
    }
  };
}

#ifndef SUNSHINE_DIRECT_RUNTIME_FIXTURE_ONLY
int main(int argc,char **argv) {
  std::setvbuf(stdout,nullptr,_IONBF,0);
  if(argc!=5 && argc!=7) {std::fputs("usage: streamline_direct_runtime <official.dll> <frozenShaders> <test.addon64> <fresh-output> [width height]\n",stderr);return 2;}
  std::thread([]{Sleep(300000);std::fputs("FAIL direct Streamline runtime watchdog\n",stderr);TerminateProcess(GetCurrentProcess(),124);}).detach();
  try {
    width=argc==7?unsigned(std::stoul(argv[5])):3840;height=argc==7?unsigned(std::stoul(argv[6])):2160;
    require(width>=640 && width<=3840 && height>=360 && height<=2160 && width%4==0 && height%4==0,"Invalid direct fixture dimensions");
    require(!sunshine_camera_fixture::flag("SUNSHINE_DEPTH_BIND_SWITCH_TEST"),"Direct fixture requires preservation mode 2 unset");
    select_native_boot();
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
