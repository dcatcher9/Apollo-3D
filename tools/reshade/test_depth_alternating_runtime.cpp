// SPDX-License-Identifier: GPL-3.0-only
// Actual rotation regression through native Game 3D with no installed FX:
// unchanged executable/assertions for old/new add-ons. Each Present's consumed
// depth and constants come from the test add-on's last-render query, its SBS
// from the production export ring. No depth, readiness or camera injection.
#include "test_raw_runtime_fixture.h"
#include "test_game3d_native_observation.h"

namespace {
  struct rotation_fixture:raw_runtime_fixture {
    enum class cadence { single,abc,aabb,offturn,offturn_history,missing,concurrent,interrupted,moving,auxiliary,copy_history,copy_end,copy_rescue,
      gap_double,gap_ui,gap_overlay,gap_long,gap_cleared,paused,end_discard,resume_pass };
    enum class rescue { clear,rect,pass_clear,discard,boxed };
    rescue rescue_route=rescue::clear;
    using action_t=BOOL (*)(api::effect_runtime *);
    native_game3d_observer game3d{*this};
    // What the native render of the latest Present consumed.
    sunshine_game3d::test::last_render latest;
    sunshine_depth::frame_depth captured;
    bool capture_ready=false,held=false;
    // The capture the previous Present showed (current or held): a Present
    // without its own current depth may show it again, as reused depth, only
    // within the source age of the Present on which it was current, measured
    // between the fixture's own clock readings around those Presents.
    struct shown_t {std::uint64_t source=0,id=0,frame=0,before=0,after=0;float center=0,blend=0;bool valid=false,known=false;} shown;
    std::uint64_t step_before=0,step_after=0;
    float last_raw=-1; // The raw center pixels() read from this Present's captured depth.
    unsigned held_presents=0;
    // This Present erased the scene's depth and drew no depth (gap_cleared, paused).
    bool erased=false;
    unsigned erase_count=0;
    // This Present's scene depth pass ended with a DISCARD and nothing wrote
    // the depth again before the Present (end_discard).
    bool end_discarded=false;
    // The list whose last pass suspends the scene's pass (resume_pass).
    com_ptr<ID3D12CommandAllocator> suspend_allocator;com_ptr<ID3D12GraphicsCommandList> suspend_list;
    // A second hidden window of the game, presenting colour only after every
    // Present of the game's window (check_second_swapchain): its own swapchain
    // and so its own effect runtime.
    HWND second_window=nullptr;
    com_ptr<IDXGISwapChain3> second_chain;com_ptr<ID3D12Resource> second_buffers[2];
    api::effect_runtime *main_runtime=nullptr,*second_runtime=nullptr;
    BOOL (*second_last)(api::effect_runtime *,sunshine_game3d::test::last_render *)=nullptr;
    std::uint64_t second_sequence=0;unsigned second_renders=0;
    // second_between: on each Present that erases the scene's depth, the
    // second window presents between that depth work and the game window's
    // Present, so its Present takes the depth frame (presented_between).
    bool second_between=false,presented_between=false;
    std::unique_ptr<target_t> overlay; // A HUD pass with its own full-size depth every Present (gap_overlay).
    com_ptr<ID3D12PipelineState> color_pipeline;com_ptr<ID3D12Resource> ui_color;com_ptr<ID3D12DescriptorHeap> ui_rtv;
    std::uint64_t rendered_sequence=0;
    // check_overlap's concurrent phase: on each Present that shows A's own
    // current depth, which members hold an end-of-frame copy of their own
    // (the test add-on's capture ticket query).
    BOOL (*capture_ticket)(api::effect_runtime *,std::uint64_t,std::uint64_t *)=nullptr;
    // Its peer-crop phase: a member's layout epoch (the test add-on's query).
    BOOL (*layout_epoch)(api::effect_runtime *,std::uint64_t,std::uint64_t *,BOOL *)=nullptr;
    std::array<unsigned,3> copied_members{};unsigned anchor_presents=0;
    bool ready() const {return latest.rendered && latest.parameters.depth_ready && latest.parameters.camera_ready;}
    float gain() const {return latest.parameters.depth_scale;}
    float zero() const {return latest.parameters.convergence[1];}
    float blend() const {return latest.parameters.strength_blend;}
    std::unique_ptr<target_t> third;
    std::vector<std::unique_ptr<target_t>> unrelated;
    cadence mode=cadence::single;
    unsigned single_role=1,phase_present=0,drawn=0,marker_sequence=0;
    unsigned raster_mask=0;
    std::uint64_t movement_started=0;
    unsigned copied_frames=0;
    bool drift=false;
    std::uint16_t marker=0x3400;
    std::array<std::uint64_t,3> lifetimes{},layouts{};
    std::array<std::array<std::uint32_t,4>,3> crops{};
    std::array<unsigned,3> stereo_observed{};
    std::ofstream trace,summary;
    com_ptr<ID3D12Resource> pixel_readback;
    struct point {unsigned x,y;};
    std::array<point,32> points{};
    static constexpr unsigned count=32,slots=count*2+1;
    float eye_difference=0;

    target_t &member(unsigned role){return role==1 ? *scene : role==2 ? *decoy : *third;}
    static float center(unsigned pattern){
      if(pattern==18)return .0625f;
      if(pattern==19)return .5f;
      if(pattern>=100 && pattern<=228)return .125f+float(pattern-100)/1024.f;
      return pattern==15 ? .25f : .125f;
    }
    // docs/reshade-sbs.md, raw-depth automation, applied to a member's known
    // draw over its captured rectangle: gain L/Q from the full-image maximum
    // Q of oriented raw t and the zero at the contrast midpoint
    // b+0.5*sum(d*d)/sum(d), b the minimum and d=t-b.
    static float raw(unsigned pattern,unsigned x,unsigned y,unsigned w,unsigned h){
      const unsigned cx=std::min(31u,unsigned((float(x)+.5f)/float(w)*32.f)),cy=std::min(17u,unsigned((float(y)+.5f)/float(h)*18.f));
      const bool inside=cx>=14 && cx<18 && cy>=7 && cy<11;
      return inside ? center(pattern) : float(cx/8+1)/16.f;
    }
    std::array<double,2> basis(unsigned pattern,unsigned w,unsigned h) const {
      double minimum=1,maximum=0,sum=0,square=0;
      for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x){const double t=raw(pattern,x,y,w,h);minimum=std::min(minimum,t);maximum=std::max(maximum,t);}
      for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x){const double d=raw(pattern,x,y,w,h)-minimum;sum+=d;square+=d*d;}
      return {limit()/maximum,minimum+.5*square/sum};
    }
    double limit() const {
      const double c=(double(height)/2160.*100.)/double(width);
      return std::min({std::min(1.5,.04/c),std::min(2.5,.04/c),.01/c})/.05;
    }
    static bool close(double actual,double expected){
      return std::isfinite(actual) && std::abs(actual-expected)<=1e-5*std::max(1.,std::abs(expected));
    }
    std::map<std::array<unsigned,3>,std::array<double,2>> bases;
    const std::array<double,2> &expected_basis(unsigned pattern,unsigned w,unsigned h){
      const std::array<unsigned,3> key{pattern,w,h};
      auto found=bases.find(key);
      if(found==bases.end())found=bases.emplace(key,basis(pattern,w,h)).first;
      return found->second;
    }
    void set_mode(cadence next,unsigned role=1){mode=next;single_role=role;phase_present=0;}
    void draw_auxiliary_scene(){
      // A non-scene pass uses the same allocation with a changing valid larger
      // viewport. Its later full clear is followed by the actual half-view
      // scene and restarts the crop, so the end-of-frame copy's crop and
      // layout are the scene's, not the larger pass's. Every other frame that
      // clear is a render pass whose depth access is CLEAR (ReShade reports it
      // as a clear), so both routes restart the crop alike.
      auto &target=*scene;
      const auto dsv=target.heap->GetCPUDescriptorHandleForHeapStart();
      commands->OMSetRenderTargets(0,nullptr,FALSE,&dsv);
      commands->ClearDepthStencilView(dsv,D3D12_CLEAR_FLAG_DEPTH,0.f,0,0,nullptr);
      commands->SetGraphicsRootSignature(root.p);
      commands->SetPipelineState(depth32_pipeline.p);
      commands->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      const unsigned factor=phase_present%2 ? 4 : 3;
      const D3D12_VIEWPORT viewport{0,0,float(width*factor)/4,float(height*factor)/4,0,1};
      const D3D12_RECT scissor{0,0,LONG(width),LONG(height)};
      commands->RSSetViewports(1,&viewport);commands->RSSetScissorRects(1,&scissor);
      struct {float w,h;unsigned pattern;} constants{viewport.Width,viewport.Height,18};
      commands->SetGraphicsRoot32BitConstants(0,3,&constants,0);
      commands->DrawInstanced(3,1,0,0);
      if(phase_present%2){
        com_ptr<ID3D12GraphicsCommandList4> list4;
        checked(commands->QueryInterface(IID_PPV_ARGS(list4.put())),"The game's list lacks render passes");
        D3D12_RENDER_PASS_DEPTH_STENCIL_DESC pass{};
        pass.cpuDescriptor=dsv;
        pass.DepthBeginningAccess.Type=D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_CLEAR;
        pass.DepthBeginningAccess.Clear.ClearValue.Format=target.format;
        pass.DepthEndingAccess.Type=D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_PRESERVE;
        pass.StencilBeginningAccess.Type=D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_NO_ACCESS;
        pass.StencilEndingAccess.Type=D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_NO_ACCESS;
        list4->BeginRenderPass(0,nullptr,&pass,D3D12_RENDER_PASS_FLAG_NONE);
        list4->EndRenderPass();
        commands->OMSetRenderTargets(0,nullptr,FALSE,&dsv);
      }else commands->ClearDepthStencilView(dsv,D3D12_CLEAR_FLAG_DEPTH,0.f,0,0,nullptr);
      const D3D12_VIEWPORT captured_view{0,0,float(width/2),float(height/2),0,1};
      const D3D12_RECT captured_scissor{0,0,LONG(width/2),LONG(height/2)};
      commands->RSSetViewports(1,&captured_view);commands->RSSetScissorRects(1,&captured_scissor);
      constants={captured_view.Width,captured_view.Height,target.pattern};
      commands->SetGraphicsRoot32BitConstants(0,3,&constants,0);
      for(unsigned i=0;i<4;++i)commands->DrawInstanced(3,1,0,0);
      commands->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    }
    void copy_depth(target_t &from,target_t &to){
      transition(commands.p,from.texture.p,D3D12_RESOURCE_STATE_DEPTH_WRITE,D3D12_RESOURCE_STATE_COPY_SOURCE);
      transition(commands.p,to.texture.p,D3D12_RESOURCE_STATE_DEPTH_WRITE,D3D12_RESOURCE_STATE_COPY_DEST);
      commands->CopyResource(to.texture.p,from.texture.p);
      transition(commands.p,to.texture.p,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_DEPTH_WRITE);
      transition(commands.p,from.texture.p,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_DEPTH_WRITE);
    }
    // A screen without the scene (a pause, inventory or loading screen, a
    // video) that keeps erasing the scene's depth and draws none: by
    // ClearDepthStencilView, a render pass whose depth access is CLEAR, one
    // that DISCARDs it, or ClearDepthStencilView limited to one rectangle that
    // spans the whole view or to one over its top-left quarter, in turn.
    void erase_scene(unsigned route){
      auto &target=*scene;const auto dsv=target.heap->GetCPUDescriptorHandleForHeapStart();
      if(route==0){commands->ClearDepthStencilView(dsv,D3D12_CLEAR_FLAG_DEPTH,target.clear_depth,0,0,nullptr);return;}
      if(route>=3){
        const D3D12_RECT rect{0,0,LONG(route==3 ? target.width : target.width/2),LONG(route==3 ? target.height : target.height/2)};
        commands->ClearDepthStencilView(dsv,D3D12_CLEAR_FLAG_DEPTH,target.clear_depth,0,1,&rect);return;
      }
      com_ptr<ID3D12GraphicsCommandList4> list4;
      checked(commands->QueryInterface(IID_PPV_ARGS(list4.put())),"The game's list lacks render passes");
      D3D12_RENDER_PASS_DEPTH_STENCIL_DESC pass{};
      pass.cpuDescriptor=dsv;
      pass.DepthBeginningAccess.Type=route==1 ? D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_CLEAR : D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_DISCARD;
      pass.DepthBeginningAccess.Clear.ClearValue.Format=target.format;
      pass.DepthBeginningAccess.Clear.ClearValue.DepthStencil.Depth=target.clear_depth;
      pass.DepthEndingAccess.Type=D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_PRESERVE;
      pass.StencilBeginningAccess.Type=D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_NO_ACCESS;
      pass.StencilEndingAccess.Type=D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_NO_ACCESS;
      list4->BeginRenderPass(0,nullptr,&pass,D3D12_RENDER_PASS_FLAG_NONE);
      list4->EndRenderPass();
    }
    void clear_depth(target_t &target){
      const auto dsv=target.heap->GetCPUDescriptorHandleForHeapStart();
      commands->ClearDepthStencilView(dsv,D3D12_CLEAR_FLAG_DEPTH,target.clear_depth,0,0,nullptr);
    }
    // copy_depth by CopyTextureRegion with an explicit source box spanning the
    // whole subresource (legal for depth-stencil, which copies only whole
    // subresources).
    void copy_depth_box(target_t &from,target_t &to){
      transition(commands.p,from.texture.p,D3D12_RESOURCE_STATE_DEPTH_WRITE,D3D12_RESOURCE_STATE_COPY_SOURCE);
      transition(commands.p,to.texture.p,D3D12_RESOURCE_STATE_DEPTH_WRITE,D3D12_RESOURCE_STATE_COPY_DEST);
      D3D12_TEXTURE_COPY_LOCATION source{},dest{};
      source.pResource=from.texture.p;source.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      dest.pResource=to.texture.p;dest.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      const D3D12_BOX box{0,0,0,UINT(from.width),UINT(from.height),1};
      commands->CopyTextureRegion(&dest,0,0,0,&source,&box);
      transition(commands.p,to.texture.p,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_DEPTH_WRITE);
      transition(commands.p,from.texture.p,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_DEPTH_WRITE);
    }
    // A game that copies its scene depth into another depth-stencil allocation
    // of the same geometry, allocated before the scene (the lower lifetime
    // identity). copy_history: a history copy at the start of the frame, before
    // the scene draws (the destination holds the previous frame), with the
    // history cleared on its first six frames as an engine reset does.
    // copy_end: the same copy after the scene draw. copy_rescue: the scene is
    // copied out, then its own allocation is cleared and reused for a small
    // first-person pass, so only the destination holds this frame's scene.
    // rescue_route: how the scene's allocation is overwritten after the copy,
    // by each route ReShade reports (ClearDepthStencilView without or with one
    // full-view rectangle, a render pass whose depth access is CLEAR, or one
    // that DISCARDs it), or the copy is a CopyTextureRegion with a
    // full-extent source box (boxed; then a plain clear).
    // Every frame draws a new center depth, so a copy one frame old shows in
    // the raw center of the captured backup (pixels()).
    // A colour-only UI pass (no depth-stencil) and a HUD pass into its own
    // full-size depth-stencil (a flat .5 in a small scissor): Presents with and
    // without other depth work and no scene depth of their own.
    void gap_pipelines(){
      const char *src=R"(
float4 vs(uint id:SV_VertexID):SV_Position{float2 uv=float2((id<<1)&2,id&2);return float4(uv*float2(2,-2)+float2(-1,1),0,1);}
float4 ps(float4 p:SV_Position):SV_Target{return float4(1,0,1,1);}
)";
      com_ptr<ID3DBlob> vs,ps,errors;
      checked(D3DCompile(src,std::strlen(src),nullptr,nullptr,nullptr,"vs","vs_5_0",0,0,vs.put(),errors.put()),"Compile UI vertex shader");
      checked(D3DCompile(src,std::strlen(src),nullptr,nullptr,nullptr,"ps","ps_5_0",0,0,ps.put(),errors.put()),"Compile UI pixel shader");
      D3D12_GRAPHICS_PIPELINE_STATE_DESC p{};
      p.pRootSignature=root.p;p.VS={vs->GetBufferPointer(),vs->GetBufferSize()};p.PS={ps->GetBufferPointer(),ps->GetBufferSize()};
      p.SampleMask=UINT_MAX;p.RasterizerState.FillMode=D3D12_FILL_MODE_SOLID;p.RasterizerState.CullMode=D3D12_CULL_MODE_NONE;
      p.RasterizerState.DepthClipEnable=TRUE;p.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;p.SampleDesc.Count=1;
      p.NumRenderTargets=1;p.RTVFormats[0]=DXGI_FORMAT_R8G8B8A8_UNORM;
      checked(game->CreateGraphicsPipelineState(&p,IID_PPV_ARGS(color_pipeline.put())),"Create colour-only UI pipeline");
      const auto heap=heap_properties(D3D12_HEAP_TYPE_DEFAULT);
      D3D12_RESOURCE_DESC d{};d.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;d.Width=width;d.Height=height;
      d.DepthOrArraySize=d.MipLevels=d.SampleDesc.Count=1;d.Format=DXGI_FORMAT_R8G8B8A8_UNORM;d.Flags=D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
      checked(game->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&d,D3D12_RESOURCE_STATE_RENDER_TARGET,nullptr,IID_PPV_ARGS(ui_color.put())),"Create UI colour target");
      D3D12_DESCRIPTOR_HEAP_DESC h{};h.Type=D3D12_DESCRIPTOR_HEAP_TYPE_RTV;h.NumDescriptors=1;
      checked(game->CreateDescriptorHeap(&h,IID_PPV_ARGS(ui_rtv.put())),"Create UI view heap");
      game->CreateRenderTargetView(ui_color.p,nullptr,ui_rtv->GetCPUDescriptorHandleForHeapStart());
    }
    void ui_color_pass(){
      const auto rtv=ui_rtv->GetCPUDescriptorHandleForHeapStart();
      commands->OMSetRenderTargets(1,&rtv,FALSE,nullptr);
      commands->SetGraphicsRootSignature(root.p);commands->SetPipelineState(color_pipeline.p);
      commands->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      const D3D12_VIEWPORT v{0,0,float(width),float(height),0,1};const D3D12_RECT s{0,0,LONG(width/4),LONG(height/8)};
      commands->RSSetViewports(1,&v);commands->RSSetScissorRects(1,&s);
      commands->DrawInstanced(3,1,0,0);
      commands->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    }
    void overlay_pass(){
      auto &t=*overlay;const auto dsv=t.heap->GetCPUDescriptorHandleForHeapStart();
      commands->OMSetRenderTargets(0,nullptr,FALSE,&dsv);
      commands->ClearDepthStencilView(dsv,D3D12_CLEAR_FLAG_DEPTH,0.f,0,0,nullptr);
      commands->SetGraphicsRootSignature(root.p);commands->SetPipelineState(depth32_pipeline.p);
      commands->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      const D3D12_VIEWPORT v{0,0,float(width),float(height),0,1};const D3D12_RECT s{0,LONG(height*7/8),LONG(width/4),LONG(height)};
      commands->RSSetViewports(1,&v);commands->RSSetScissorRects(1,&s);
      struct{float w,h;unsigned pattern;}c{float(width),float(height),19};
      commands->SetGraphicsRoot32BitConstants(0,3,&c,0);commands->DrawInstanced(3,1,0,0);
      commands->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    }
    void draw_copy_case(){
      const unsigned pattern=100+8*(copied_frames%16);
      scene->pattern=decoy->pattern=pattern;
      if(mode==cadence::copy_history){
        if(copied_frames<6)clear_depth(*decoy);
        copy_depth(*scene,*decoy);draw(*scene);
      }else if(mode==cadence::copy_end){draw(*scene);copy_depth(*scene,*decoy);}
      else{
        draw(*scene);
        if(rescue_route==rescue::boxed)copy_depth_box(*scene,*decoy);else copy_depth(*scene,*decoy);
        const auto dsv=scene->heap->GetCPUDescriptorHandleForHeapStart();
        if(rescue_route==rescue::pass_clear)erase_scene(1);
        else if(rescue_route==rescue::discard)erase_scene(2);
        else if(rescue_route==rescue::rect){
          const D3D12_RECT all{0,0,LONG(width),LONG(height)};
          commands->ClearDepthStencilView(dsv,D3D12_CLEAR_FLAG_DEPTH,scene->clear_depth,0,1,&all);
        }else clear_depth(*scene);
        commands->OMSetRenderTargets(0,nullptr,FALSE,&dsv);
        commands->SetGraphicsRootSignature(root.p);commands->SetPipelineState(depth32_pipeline.p);
        commands->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        const D3D12_VIEWPORT viewport{0,0,float(width),float(height),0,1};
        if(rescue_route==rescue::discard){
          // A DISCARD leaves the depth undefined: the pass's full-view reset
          // draw (a flat .5) replaces it, so only the destination holds the
          // scene whatever the driver kept.
          const D3D12_RECT all{0,0,LONG(width),LONG(height)};
          commands->RSSetViewports(1,&viewport);commands->RSSetScissorRects(1,&all);
          struct {float w,h;unsigned pattern;} reset{float(width),float(height),19};
          commands->SetGraphicsRoot32BitConstants(0,3,&reset,0);
          commands->DrawInstanced(3,1,0,0);
        }
        const D3D12_RECT scissor{LONG(width*3/4),LONG(height*3/4),LONG(width),LONG(height)};
        commands->RSSetViewports(1,&viewport);commands->RSSetScissorRects(1,&scissor);
        struct {float w,h;unsigned pattern;} constants{float(width),float(height),19};
        commands->SetGraphicsRoot32BitConstants(0,3,&constants,0);
        commands->DrawInstanced(3,1,0,0);
        commands->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
      }
      ++copied_frames;
    }
    // The scene's depth pass as one D3D12 render pass: its depth CLEARed at the
    // start, the scene drawn inside over the top-left 1/divisor of each side
    // (none for 0), and its depth ending access `end`; recorded into `into`
    // (else the frame's list) with `flags`.
    void scene_pass(D3D12_RENDER_PASS_ENDING_ACCESS_TYPE end,ID3D12GraphicsCommandList *into=nullptr,
        D3D12_RENDER_PASS_FLAGS flags=D3D12_RENDER_PASS_FLAG_NONE,unsigned divisor=1){
      auto &target=*scene;
      auto *const list=into ? into : commands.p;
      com_ptr<ID3D12GraphicsCommandList4> list4;
      checked(list->QueryInterface(IID_PPV_ARGS(list4.put())),"The game's list lacks render passes");
      D3D12_RENDER_PASS_DEPTH_STENCIL_DESC pass{};
      pass.cpuDescriptor=target.heap->GetCPUDescriptorHandleForHeapStart();
      pass.DepthBeginningAccess.Type=D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_CLEAR;
      pass.DepthBeginningAccess.Clear.ClearValue.Format=target.format;
      pass.DepthBeginningAccess.Clear.ClearValue.DepthStencil.Depth=target.clear_depth;
      pass.DepthEndingAccess.Type=end;
      pass.StencilBeginningAccess.Type=D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_NO_ACCESS;
      pass.StencilEndingAccess.Type=D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_NO_ACCESS;
      list4->BeginRenderPass(0,nullptr,&pass,flags);
      if(divisor){
        list->SetGraphicsRootSignature(root.p);list->SetPipelineState(depth32_pipeline.p);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        const D3D12_VIEWPORT viewport{0,0,float(target.width/divisor),float(target.height/divisor),0,1};
        const D3D12_RECT scissor{0,0,LONG(target.width),LONG(target.height)};
        list->RSSetViewports(1,&viewport);list->RSSetScissorRects(1,&scissor);
        struct {float w,h;unsigned pattern;} constants{viewport.Width,viewport.Height,target.pattern};
        list->SetGraphicsRoot32BitConstants(0,3,&constants,0);
        for(unsigned i=0;i<target.draws;++i)list->DrawInstanced(3,1,0,0);
      }
      list4->EndRenderPass();
    }
    // end_discard: the scene's pass DISCARDs its depth at the end every
    // Present, as a game does whose later work never reads that depth. Every
    // other Present that pass is the frame's last access to the scene's depth,
    // which is then undefined when the frame ends; on the Presents between, a
    // second pass (its CLEAR, the scene drawn again, PRESERVE) writes it again.
    // A new center depth every frame, so a copy one frame old shows in the raw
    // center (pixels()).
    void draw_end_discard(){
      scene->pattern=100+8*(phase_present%16);
      scene_pass(D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_DISCARD);
      end_discarded=phase_present%2;
      if(!end_discarded)scene_pass(D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_PRESERVE);
    }
    // resume_pass: the scene's pass split across two command lists executed
    // in one ExecuteCommandLists call, as a renderer records one pass on
    // several threads: the scene drawn in a pass that CLEARs its depth,
    // PRESERVEs it and suspends, the last thing in its own list, and the
    // frame's list beginning with the pass that resumes it (the same accesses,
    // as the pair requires) and draws the top-left quarter again (a later
    // part of the same pass). A new center depth every frame.
    void draw_resume_pass(){
      if(!suspend_list.p){
        checked(game->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(suspend_allocator.put())),"Create the suspending list's allocator");
        checked(game->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,suspend_allocator.p,nullptr,IID_PPV_ARGS(suspend_list.put())),"Create the suspending list");
      }else{
        checked(suspend_allocator->Reset(),"Reset the suspending list's allocator");
        checked(suspend_list->Reset(suspend_allocator.p,nullptr),"Reset the suspending list");
      }
      scene->pattern=100+8*(phase_present%16);
      scene_pass(D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_PRESERVE,suspend_list.p,D3D12_RENDER_PASS_FLAG_SUSPENDING_PASS);
      checked(suspend_list->Close(),"Close the suspending list");
      execute_first=suspend_list.p;
      scene_pass(D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_PRESERVE,commands.p,D3D12_RENDER_PASS_FLAG_RESUMING_PASS,2);
    }
    void draw_depth(){
      drawn=0;raster_mask=0;end_discarded=false;
      if(mode==cadence::single)drawn=single_role;
      else if(mode==cadence::abc || mode==cadence::offturn || mode==cadence::offturn_history)drawn=phase_present%3+1;
      else if(mode==cadence::copy_history || mode==cadence::copy_end || mode==cadence::end_discard ||
        mode==cadence::resume_pass)drawn=1;
      else if(mode==cadence::copy_rescue)drawn=2; // The destination holds this frame's scene.
      else if(mode==cadence::aabb)drawn=(phase_present/2)%2+1;
      else if(mode==cadence::concurrent || mode==cadence::moving || mode==cadence::auxiliary)drawn=1;
      else if(mode==cadence::interrupted || mode==cadence::gap_double || mode==cadence::gap_ui || mode==cadence::gap_overlay ||
        mode==cadence::gap_cleared)
        drawn=phase_present%2 ? 0 : 1;
      else if(mode==cadence::gap_long)drawn=phase_present%40 ? 0 : 1;
      if(mode==cadence::gap_overlay)overlay_pass();
      if(mode==cadence::gap_ui && !drawn)ui_color_pass();
      erased=(mode==cadence::gap_cleared || mode==cadence::paused) && !drawn;
      if(erased)erase_scene(erase_count++%5);
      if(erased && second_between && second_chain.p){
        // Submit the game's work so far, present the second window, then go
        // on recording the game window's frame.
        submit();present_second();begin_commands();presented_between=true;
      }
      if(mode==cadence::moving){
        if(!movement_started)movement_started=clock.now();
        // Looking repeatedly between coherent near/middle/far center patches
        // begins before the first effect frame. No stationary calibration lead-in.
        constexpr unsigned patterns[]{18,14,15};
        scene->pattern=patterns[((clock.now()-movement_started)/300)%3];
      }
      if(drift)for(unsigned r=1;r<=3;++r)member(r).pattern=100+std::min(128u,phase_present/6);
      ++phase_present;
      if(!drawn)return;
      raster_mask=1u<<(drawn-1);
      if(mode==cadence::concurrent){
        // A was qualified first and all peers have identical quality. Vary
        // command order so the stable preferred source cannot be a last-draw rule.
        for(unsigned i=0;i<3;++i)draw(member((i+phase_present)%3+1));
        raster_mask=7;return;
      }
      if(mode==cadence::auxiliary){draw_auxiliary_scene();return;}
      if(mode==cadence::copy_history || mode==cadence::copy_end || mode==cadence::copy_rescue){
        raster_mask=1;draw_copy_case();return;
      }
      if(mode==cadence::end_discard){draw_end_discard();return;}
      if(mode==cadence::resume_pass){draw_resume_pass();return;}
      const auto draw_unrelated=[&]{
        for(unsigned i=0;i<unrelated.size();++i)
          draw(*unrelated[(i+phase_present)%unrelated.size()]);
      };
      if(!unrelated.empty() && phase_present%2)draw_unrelated();
      auto &current=member(drawn);
      // Off-turn transfers into another live member of the same geometry: a
      // copy of the drawn member after its draw (offturn) or, as a history
      // buffer is filled, of its previous contents before its draw
      // (offturn_history). Neither is depth of the destination's own: the
      // drawn member stays the only current copy of this frame on every Present.
      if(mode==cadence::offturn_history)copy_depth(current,member(drawn%3+1));
      draw(current);
      if(mode==cadence::offturn)copy_depth(current,member(drawn%3+1));
      if(!unrelated.empty() && !(phase_present%2))draw_unrelated();
    }
    void update_marker(){
      // Exact half values; period1024 is coprime to 3. Successive markers differ
      // by17 half codes, making ordinary1/2/3-frame stale outputs distinguishable.
      marker=std::uint16_t(0x3400+(++marker_sequence*17)%1024);
      void *mapped=nullptr;const D3D12_RANGE no_read{0,0};
      checked(source_upload->Map(0,&no_read,&mapped),"Map actual source marker");
      for(unsigned y=0;y<height;++y){
        auto *row=reinterpret_cast<std::uint16_t *>(static_cast<std::uint8_t *>(mapped)+source_footprint.Offset+size_t(y)*source_footprint.Footprint.RowPitch);
        for(unsigned x=0;x<width;++x)row[x*4+1]=marker;
      }
      source_upload->Unmap(0,nullptr);
    }
    bool current(unsigned role){
      const auto &m=member(role);
      return capture_ready &&
        captured.source_resource.handle==reinterpret_cast<std::uint64_t>(m.texture.p) &&
        captured.width==m.width && captured.height==m.height;
    }
    bool full(){return drawn && current(drawn) && ready() && blend()==1.f;}
    void require_basis(){
      require(drawn && current(drawn),"No actual current member for basis assertion");
      const auto &expected=expected_basis(member(drawn).pattern,captured.active_width,captured.active_height);
      require(close(gain(),expected[0]) && close(zero(),expected[1]),
        "Physical member inherited another member's gain/zero instead of its own range and contrast midpoint");
    }
    void copy_pixel(ID3D12Resource *texture,unsigned x,unsigned y,unsigned slot,DXGI_FORMAT format){
      D3D12_TEXTURE_COPY_LOCATION from{},to{};
      from.pResource=texture;from.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      to.pResource=pixel_readback.p;to.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
      to.PlacedFootprint.Offset=UINT64(slot)*D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT;
      to.PlacedFootprint.Footprint={format,1,1,1,D3D12_TEXTURE_DATA_PITCH_ALIGNMENT};
      const D3D12_BOX box{x,y,0,x+1,y+1,1};
      commands->CopyTextureRegion(&to,0,0,0,&from,&box);
    }
    void pixels(){
      // Compact actual native readbacks each present, no fitted registration:
      // this Present's exported SBS and the depth its render consumed.
      const bool direct_flat=!capture_ready && drawn && member(drawn).pattern==19;
      auto *depth_resource=capture_ready ? reinterpret_cast<ID3D12Resource *>(captured.resource.handle) :
        direct_flat ? member(drawn).texture.p : nullptr;
      game3d.read_exported([&](ID3D12Resource *sbs){
        begin_commands();
        transition(commands.p,sbs,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_SOURCE);
        for(unsigned i=0;i<count;++i){
          copy_pixel(sbs,points[i].x,points[i].y,i*2,DXGI_FORMAT_R16G16B16A16_FLOAT);
          copy_pixel(sbs,width+points[i].x,points[i].y,i*2+1,DXGI_FORMAT_R16G16B16A16_FLOAT);
        }
        transition(commands.p,sbs,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_COMMON);
        // The flat-startup oracle also reads the original game allocation when
        // the selector correctly rejects its copy. A configured draw alone is
        // not evidence that the real depth image contains the intended .5.
        if(depth_resource){
          const auto before=direct_flat ? D3D12_RESOURCE_STATE_DEPTH_WRITE : D3D12_RESOURCE_STATE_COPY_DEST;
          transition(commands.p,depth_resource,before,D3D12_RESOURCE_STATE_COPY_SOURCE);
          copy_pixel(depth_resource,direct_flat ? member(drawn).width/2 : captured.x+captured.active_width/2,
            direct_flat ? member(drawn).height/2 : captured.y+captured.active_height/2,slots-1,DXGI_FORMAT_R32_TYPELESS);
          transition(commands.p,depth_resource,D3D12_RESOURCE_STATE_COPY_SOURCE,before);
        }
        submit();
      });
      void *mapped=nullptr;const D3D12_RANGE range{0,size_t(slots)*512};
      checked(pixel_readback->Map(0,&range,&mapped),"Map current native pixel probes");
      const auto *bytes=static_cast<const std::uint8_t *>(mapped);
      const auto value=[&](unsigned slot,unsigned c){std::uint16_t v=0;std::memcpy(&v,bytes+size_t(slot)*512+c*2,2);return half_float(v);};
      eye_difference=0;
      for(unsigned i=0;i<count;++i){
        const auto base=(size_t(points[i].y)*width+points[i].x)*8;
        for(unsigned c=0;c<4;++c){
          std::uint16_t code=0;std::memcpy(&code,source_bytes.data()+base+c*2,2);
          const float expected=c==1 ? half_float(marker) : half_float(code);
          const float left=value(i*2,c),right=value(i*2+1,c);
          require(std::isfinite(left)&&std::isfinite(right),"Current native stereo/mono has nonfinite RGBA");
          if(c==1)require(std::abs(left-expected)<.001f && std::abs(right-expected)<.001f,"SBS carries a stale/wrong current-frame color marker");
          if(!ready())require(std::abs(left-expected)<.003f && std::abs(right-expected)<.003f,"Unavailable depth did not show actual current mono color");
          if(c==0 || c==2)eye_difference=std::max(eye_difference,std::abs(left-right));
        }
      }
      if(depth_resource){
        float raw=0;std::memcpy(&raw,bytes+size_t(slots-1)*512,4);last_raw=raw;
        if(held){if(shown.known)require(raw==shown.center,"Held backup no longer holds the raw center depth it showed");}
        else require(drawn && raw==center(member(drawn).pattern),"Current backup contains stale/wrong-member raw center depth");
      }
      const D3D12_RANGE no_write{0,0};pixel_readback->Unmap(0,&no_write);
      if(drawn && full() && eye_difference>.002f)++stereo_observed[drawn-1];
    }
    void tick(const char *phase){
      // A double present repeats the real frame's colour.
      if(!(mode==cadence::gap_double && phase_present%2))update_marker();
      step_before=clock.now();step();step_after=clock.now();game3d.no_effects();
      latest=game3d.last_render();
      require(latest.sequence>rendered_sequence,"A Present had no native Game 3D render to observe");
      rendered_sequence=latest.sequence;
      captured=latest.depth;capture_ready=captured.ready;held=capture_ready && captured.reused_depth;
      if(held && shown.known){
        // Only the capture the previous Present showed, its frame unchanged,
        // within the source age, with the scene (and ramp) it was shown with.
        // (The first tick follows Presents this fixture did not observe.)
        if(!(shown.valid && captured.source_resource.handle==shown.source && captured.source_id==shown.id && captured.frame_index==shown.frame))
          std::fprintf(stderr,"DIAG held source=%llx id=%llu frame=%llu; shown valid=%d source=%llx id=%llu frame=%llu; drawn=%u\n",
            (unsigned long long)captured.source_resource.handle,(unsigned long long)captured.source_id,(unsigned long long)captured.frame_index,
            shown.valid?1:0,(unsigned long long)shown.source,(unsigned long long)shown.id,(unsigned long long)shown.frame,drawn);
        require(shown.valid && captured.source_resource.handle==shown.source && captured.source_id==shown.id &&
          captured.frame_index==shown.frame,"A held Present showed another capture than the previous Present's");
        require(step_before-shown.after<250,"Depth was held beyond the source age");
        require(blend()==shown.blend,"A held Present advanced or reset the strength ramp");
      }
      // A frame whose last access to the scene's depth DISCARDed it leaves
      // undefined depth: never shown as that Present's current depth.
      if(end_discarded)require(!capture_ready || held,
        "A Present whose scene depth its render pass DISCARDed showed that frame's undefined depth as current");
      if(held)++held_presents;
      else{
        require(!capture_ready || (drawn && current(drawn)),"Wrong/old physical depth admitted for this rendered present");
        // Nothing here invalidates the shown capture: a Present with no
        // current depth within its source age must show it again, unless it
        // erased the scene's depth and drew none (below).
        require(capture_ready || drawn || erased || !shown.valid || step_after-shown.before>=250,
          "A Present without depth within the source age of the previous Present's depth was flat");
      }
      // A screen without the scene holds nothing: mono from its first Present.
      if(erased)require(!held && !capture_ready && !ready() && blend()==0.f,
        "A Present that erased the scene's depth and drew none held or showed depth");
      require(!ready() || capture_ready,"Camera ready without current legally preserved depth");
      require(drawn || held || (!capture_ready && !ready() && blend()==0.f),"Color-only Present beyond the source age reused depth or stereo");
      if(capture_ready && !held){
        lifetimes[drawn-1]=captured.source_id;layouts[drawn-1]=captured.layout_epoch;
        crops[drawn-1]={captured.x,captured.y,captured.active_width,captured.active_height};
      }
      pixels();
      if(capture_ticket && capture_ready && !held && current(1)){
        for(unsigned role=1;role<=3;++role){
          std::uint64_t id=0;
          require(capture_ticket(observed.runtime,reinterpret_cast<std::uint64_t>(member(role).texture.p),&id),"The capture ticket query failed");
          if(id)++copied_members[role-1];
        }
        ++anchor_presents;
      }
      if(capture_ready && !held)shown={captured.source_resource.handle,captured.source_id,captured.frame_index,step_before,step_after,
        center(member(drawn).pattern),blend(),true};
      else if(held && !shown.known)
        // A hold of a Present before the first observed one: its current
        // Present ended before this one began (no earlier bound is known).
        shown={captured.source_resource.handle,captured.source_id,captured.frame_index,0,step_before,last_raw,blend(),true};
      else if(!capture_ready)shown.valid=false;
      shown.known=true;
      trace<<phase<<','<<clock.now()<<','<<latest.sequence<<','<<drawn<<','
        <<(drawn?reinterpret_cast<std::uint64_t>(member(drawn).texture.p):0)<<','<<captured.source_resource.handle<<','
        <<captured.source_id<<','<<captured.layout_epoch<<','<<captured.frame_index<<','<<capture_ready<<','<<ready()<<','
        <<gain()<<','<<zero()<<','<<blend()<<','
        <<marker<<','<<eye_difference<<','<<(drawn?center(member(drawn).pattern):0.f)<<','
        <<captured.x<<','<<captured.y<<','<<captured.active_width<<','<<captured.active_height<<','<<raster_mask<<','<<held<<'\n';
      require(trace.good(),"Cannot record actual-frame rotation trajectory");
      if(second_chain.p && !presented_between)present_second();
      presented_between=false;
    }
    void warm(const char *phase,bool basis=true,const std::function<void(std::uint64_t)> &guard={}){
      const auto started=clock.now();unsigned consecutive=0;
      do{
        tick(phase);if(guard)guard(clock.now()-started);
        if(full()){if(basis)require_basis();++consecutive;}else consecutive=0;
      }while(consecutive<12 && clock.now()-started<15000);
      std::printf("MEASURE %s warmup_ms=%llu consecutive_full=%u\n",phase,static_cast<unsigned long long>(clock.now()-started),consecutive);
      require(consecutive>=12,"Current rotating sources never reached continuous camera readiness/full blend within15s");
    }
    void stable(const char *phase,unsigned frames,bool basis=true){
      std::array<unsigned,3> counts{};
      for(unsigned i=0;i<frames;++i){tick(phase);require(full(),"Current rotating member lost camera readiness/full stereo blend");if(basis)require_basis();++counts[drawn-1];}
      summary<<phase<<','<<frames<<','<<counts[0]<<','<<counts[1]<<','<<counts[2]<<",pass\n";
      summary.flush();trace.flush();
      std::printf("PASS %s frames=%u A=%u B=%u C=%u current-depth/current-color/full-stereo\n",phase,frames,counts[0],counts[1],counts[2]);
    }
    void continuous(unsigned role){
      set_mode(cadence::single,role);const auto label="continuous-control-"+std::to_string(role);
      warm(label.c_str());stable(label.c_str(),12);
    }
    // A run of colour-only Presents: the depth of the last real frame is held
    // within the source age (tick() checks every Present), then mono. Back
    // after more than the source age without depth, stereo fades in again.
    void color_only_gap(const char *phase){
      set_mode(cadence::missing);
      unsigned held_run=0,mono=0;std::uint64_t first_mono=0;
      // Mono for longer than the source age, so the ramp restarts.
      do{tick(phase);if(held)++held_run;else if(!mono++)first_mono=step_after;}while(mono<4 || step_after-first_mono<350);
      require(held_run>0,"A colour-only gap after current depth held nothing");
      std::printf("PASS %s: %u held Presents within the source age, then %u mono\n",phase,held_run,mono);
    }
    void require_fade_in(cadence next,unsigned role,const char *phase){
      set_mode(next,role);
      const auto until=clock.now()+5000;
      do tick(phase);while(!ready() && clock.now()<until);
      require(ready() && blend()<1.f,"Depth returning after more than the source age did not fade stereo in again");
    }
    void missing_then_recover(){
      color_only_gap("true-color-only-gap");
      require_fade_in(cadence::single,1,"current-depth-return");
      set_mode(cadence::single,1);warm("current-depth-return");stable("current-depth-return",20);
    }
    void check_overlap(){
      continuous(1); // Automatic selection: A qualifies before equal-quality peers.
      set_mode(cadence::abc);warm("overlap-ABC-control");stable("overlap-ABC-control",30);
      set_mode(cadence::concurrent);
      // Persistent overlap must not lock an already-qualified family into mono.
      // The expected winner A was specified before any concurrent result is read.
      warm("concurrent-anchor-recovery");
      // A shows its own current copy on every Present, so its peers B and C,
      // drawn on every Present too, are copied only when the add-on may read
      // them: a raw sample due (every 250 ms each) or the probe's sample.
      // Until 11-01 each was copied on every Present (on D3D12 a snapshot and
      // a publish copy of the whole plane apiece), A's depth unchanged.
      capture_ticket=reinterpret_cast<decltype(capture_ticket)>(GetProcAddress(GetModuleHandleW(L"SunshineSBSTest.addon64"),"SunshineDepthTestCaptureTicket"));
      require(capture_ticket,"The overlap case needs the test add-on's capture ticket query");
      copied_members={};anchor_presents=0;
      stable("concurrent-anchor",60);
      const auto counted=anchor_presents;const auto copies=copied_members;
      capture_ticket=nullptr;
      std::printf("MEASURE concurrent-anchor presents=%u copied A=%u B=%u C=%u\n",counted,copies[0],copies[1],copies[2]);
      require(counted==60 && copies[0]==counted,"A Present that showed A's current depth had no end-of-frame copy of A");
      // Raw samples and probe visits read a peer a few times a second (about
      // one Present in five at this fixture's 31 ms Presents on 11-01):
      // bounded below the anchor's every-Present copies with room for slower
      // Presents.
      require(copies[1]*4<counted*3 && copies[2]*4<counted*3,
        "A peer drawn beside the shown anchor was copied on most Presents although nothing read its copy");
      // A peer B drawn with a new crop beside the shown anchor takes that
      // crop on the Present it draws it, copied or not
      // (observe_uncopied_member, 11-01): its layout epoch advances there,
      // so its evidence, set membership and raw roster entry follow the
      // change as with a copy. Without it B's layout moved only at its next
      // copy (the fix review's fixrev-generic-1: no row changed an uncopied
      // peer's crop before). A change is judged where B was a retained set
      // member, A showed its own current copy and B was not copied on that
      // Present (its capture ticket), the case the helper exists for. A
      // member whose layout changed leaves the set, and a non-member is not
      // sampled, so after each change B returns to its full view and
      // qualifies again in the ABC rotation before the next one.
      layout_epoch=reinterpret_cast<decltype(layout_epoch)>(GetProcAddress(GetModuleHandleW(L"SunshineSBSTest.addon64"),"SunshineDepthTestLayoutEpoch"));
      const auto ticket_of=reinterpret_cast<decltype(capture_ticket)>(GetProcAddress(GetModuleHandleW(L"SunshineSBSTest.addon64"),"SunshineDepthTestCaptureTicket"));
      require(layout_epoch && ticket_of,"The overlap case needs the test add-on's layout epoch and capture ticket queries");
      const auto peer=[&](BOOL &retained){
        std::uint64_t value=0;retained=FALSE;
        require(layout_epoch(observed.runtime,reinterpret_cast<std::uint64_t>(member(2).texture.p),&value,&retained),"The layout epoch query failed");
        return value;
      };
      unsigned judged=0,followed=0,copied_changes=0,cycles=0;
      while(judged<3 && cycles<10){
        ++cycles;
        set_mode(cadence::concurrent);warm("concurrent-peer-crop-anchor");
        BOOL retained=FALSE;
        for(unsigned waited=0;waited<200;++waited){
          peer(retained);if(retained && waited>=10)break;
          tick("concurrent-peer-crop-wait");
        }
        if(retained){
          const auto before=peer(retained);
          decoy->partial_viewport=true;
          tick("concurrent-peer-crop");
          std::uint64_t copy=0;
          require(ticket_of(observed.runtime,reinterpret_cast<std::uint64_t>(member(2).texture.p),&copy),"The capture ticket query failed");
          if(capture_ready && !held && current(1)){
            if(copy)++copied_changes;
            else{++judged;followed+=peer(retained)>before;}
          }
          decoy->partial_viewport=false;
        }
        set_mode(cadence::abc);warm("concurrent-peer-crop-rotation");
      }
      layout_epoch=nullptr;
      std::printf("MEASURE concurrent-peer-crop cycles=%u judged=%u followed=%u copied_changes=%u\n",cycles,judged,followed,copied_changes);
      require(judged>=3,"The peer-crop phase judged fewer than three uncopied crop changes of a set member beside the shown anchor");
      require(followed==judged,"A peer drawn with a new crop beside the shown anchor, not copied, kept its old layout until its next copy");
      set_mode(cadence::abc);warm("concurrent-back-to-ABC");stable("concurrent-back-to-ABC",30);
      missing_then_recover();
      require(stereo_observed[0]>0,"Concurrent preferred source never produced actual stereo pixels");
      std::puts("PASS overlap: qualified A/ABC, persistent concurrent raster with current A (peers copied only when read, a peer's new crop taken on the Present it drew it), return to ABC, current mono on real gaps");
    }
    void check_interrupted_startup(){
      const auto started=clock.now();unsigned valid=0,missing=0,consecutive_valid=0;
      do{
        tick("interrupted-startup");
        if(drawn){
          ++valid;
          if(full()){require_basis();++consecutive_valid;}else consecutive_valid=0;
        }else ++missing; // tick() verifies a hold within the source age, or mono.
      }while(consecutive_valid<30 && clock.now()-started<15000);
      std::printf("MEASURE interrupted-startup elapsed_ms=%llu valid=%u missing=%u full_valid_run=%u\n",
        static_cast<unsigned long long>(clock.now()-started),valid,missing,consecutive_valid);
      require(missing>=30 && consecutive_valid==30,
        "Interleaved missing presents prevented async exact-source startup/full-strength on valid frames");
      require(stereo_observed[0]>0,"Recovered valid frames never produced actual stereo pixels");
      missing_then_recover();
      std::printf("PASS interrupted startup: actual async samples survive missing presents; valid frames full stereo, absent frames hold their real frame's depth (%u held)\n",held_presents);
    }
    void check_rotating_startup(){
      // All three allocations have rendered ABC since the very first game
      // draw. There is no single-source warmup, pin, recalibration action or
      // injected camera state before this observation window.
      warm("fresh-ABC-startup");
      stereo_observed={};
      stable("fresh-ABC-sustained",90);
      for(unsigned role=0;role<3;++role){
        require(lifetimes[role]!=0 && stereo_observed[role]>0,
          "Fresh rotating member never established its own current depth and actual stereo pixels");
        for(unsigned other=0;other<role;++other)
          require(lifetimes[role]!=lifetimes[other],"Fresh rotating resources share a lifetime identity");
      }
      std::puts("PASS rotating startup: initial ABC without prior history, each member's own range/midpoint basis, 90 current-depth/current-color/full-stereo frames");
    }
    void check_crowded_startup(){
      // Every allocation is new. Older-lifetime unrelated buffers render on
      // every present, so a serial lifetime-ID tour cannot hide behind prior
      // source qualification. Content/camera state is never injected.
      const auto started=clock.now();unsigned consecutive=0;
      do{
        tick("crowded-fresh-ABC-startup");
        if(full()){require_basis();++consecutive;}else consecutive=0;
      }while(consecutive<12 && clock.now()-started<8000);
      std::printf("MEASURE crowded-startup warmup_ms=%llu consecutive_full=%u unrelated=%zu\n",
        static_cast<unsigned long long>(clock.now()-started),consecutive,unrelated.size());
      require(consecutive==12,"Fresh rotating sources among12 unrelated buffers did not reach full stereo within8s");
      stereo_observed={};stable("crowded-ABC-sustained",90);
      for(unsigned role=0;role<3;++role){
        require(lifetimes[role]!=0 && stereo_observed[role]>0,
          "Crowded rotating member never supplied its own current depth and actual stereo pixels");
        for(unsigned other=0;other<role;++other)
          require(lifetimes[role]!=lifetimes[other],"Crowded rotating resources share a lifetime identity");
      }
      std::puts("PASS crowded startup: 12 older flat active buffers, fresh ABC member bases, off-turn copies into live members, 90 current-color/current-depth/full-stereo frames");
    }
    void check_flat_startup(){
      // A has rendered the same .5 plane since the first game draw, before
      // effects loaded. It is written depth with a known clear convention,
      // but cannot supply a useful scene reference. No UI actions are used.
      const auto original=scene->texture.p;
      const auto initial_started=clock.now();unsigned flat_frames=0;
      do{
        tick("flat-startup-mono");++flat_frames;
        require(!ready() && blend()==0.f,
          "Flat interior depth initialized a camera instead of remaining current mono");
      }while(clock.now()-initial_started<2600);
      std::printf("PASS initial flat depth: %u actual .5 draws over%llu ms, current mono throughout\n",flat_frames,
        static_cast<unsigned long long>(clock.now()-initial_started));
      scene->pattern=14;
      warm("flat-to-useful-fresh-reference");stable("flat-to-useful-sustained",30);
      const auto useful_lifetime=lifetimes[0];
      require(scene->texture.p==original && useful_lifetime && stereo_observed[0]>0,
        "Flat-to-useful phase did not reach actual stereo on its original live allocation");
      // A previously valid target may remain usable until its 1500 ms expiry.
      // Rejected flat samples must never move the gain or zero, even during
      // that interval: a ready render keeps the useful basis and the
      // controller holds its gain once the render is mono.
      const auto useful=expected_basis(14,width,height);
      scene->pattern=19;
      const auto flat_return_started=clock.now();unsigned expired_mono=0;
      do{
        tick("useful-to-flat-hold");
        if(ready())require(close(gain(),useful[0]) && close(zero(),useful[1]),
          "Rejected flat samples changed an established useful source's gain/zero");
        else require(close(game3d.automatic().scale,useful[0]),"Rejected flat samples changed the held gain");
        if(current(1))require(captured.source_id==useful_lifetime,"Same live flat source changed lifetime identity");
        if(clock.now()-flat_return_started>=1500){
          require(!ready() && blend()==0.f,
            "Flat-only evidence kept stale ready depth beyond target expiry");
          ++expired_mono;
        }
      }while(clock.now()-flat_return_started<2600);
      require(expired_mono>0,"Returning-flat phase did not exercise expired-evidence current mono");
      scene->pattern=14;stereo_observed[0]=0;
      warm("flat-return-useful-retained-reference");stable("flat-return-useful-sustained",30);
      require(scene->texture.p==original && lifetimes[0]==useful_lifetime && stereo_observed[0]>0,
        "Useful return changed lifetime or never restored actual current stereo");
      std::puts("PASS flat content admission: no flat startup, fresh range/midpoint basis on useful depth, held basis through later flat rejection and same-source stereo recovery");
    }
    void check_moving_startup(){
      const auto started=clock.now();unsigned consecutive=0,changes=0,previous=scene->pattern;
      do{
        tick("moving-center-startup");
        if(scene->pattern!=previous){++changes;previous=scene->pattern;}
        if(full()){
          // Every moving pattern keeps the nearest band at .25: gain L/.25,
          // and the zero stays within the drawn raw range.
          const float t0=zero();
          require(close(gain(),limit()/.25) && std::isfinite(t0) && t0>=.0625f && t0<=.25f,
            "Moving-source camera escaped the independently known raw range");
          ++consecutive;
        }else consecutive=0;
      }while((consecutive<60 || changes<6) && clock.now()-started<15000);
      std::printf("MEASURE moving-center-startup elapsed_ms=%llu center_changes=%u full_run=%u\n",
        static_cast<unsigned long long>(clock.now()-started),changes,consecutive);
      require(consecutive>=60 && changes>=6,"Continuous valid moving depth never reached persistent full stereo within15s");
      require(stereo_observed[0]>0,"Moving-source initialization never reached actual stereo pixels");
      color_only_gap("moving-real-gap");
      // Keep moving on return; do not hide a startup problem behind a still scene.
      set_mode(cadence::moving);warm("moving-return",false);stable("moving-return",20,false);
      std::puts("PASS moving startup: near/middle/far scene movement begins before calibration; current-depth/current-color finite stereo");
    }
    void check_layout(){
      continuous(1);
      const auto original_lifetime=lifetimes[0],full_layout=layouts[0];
      require(crops[0]==std::array<std::uint32_t,4>{0,0,width,height},"Layout control did not begin from the full captured view");
      set_mode(cadence::auxiliary);
      unsigned good=0;std::uint64_t partial_layout=0;
      const auto until=clock.now()+10000;
      do{
        tick("stable-partial-capture-varying-auxiliary");
        require(!ready(),"Unsupported partial scene crop retained a full-view camera");
        if(current(1)){
          require(captured.source_id==original_lifetime &&
            crops[0]==std::array<std::uint32_t,4>{0,0,width/2,height/2},"Expected current partial scene copy was not captured");
          if(!partial_layout){partial_layout=layouts[0];require(partial_layout!=full_layout,"A real captured-crop change did not advance its layout");}
          require(layouts[0]==partial_layout,"Unrelated auxiliary viewport activity changed the stable captured scene basis");
          ++good;
        }
      }while(good<30 && clock.now()<until);
      require(good==30,"Stable partial captured view never supplied30 current native copies");
      set_mode(cadence::single,1);
      warm("actual-full-crop-return",true,[&](std::uint64_t elapsed){
        if(current(1))require(crops[0]==std::array<std::uint32_t,4>{0,0,width,height} && layouts[0]!=partial_layout,
          "Real full-view restoration failed to change the captured basis");
        if(elapsed<650)require(!ready(),"Changed capture basis reused the previous initialized camera");
      });
      stable("actual-full-crop-return",30);missing_then_recover();
      std::puts("PASS layout: stable captured partial view ignores unrelated auxiliary activity, real crop change starts a new basis");
    }
    // The copy destination was allocated first (the lower lifetime identity).
    // The scene's own draw, or in copy_rescue the destination holding the pass
    // its source cleared away, must be the captured current depth on every
    // Present (tick, pixels: the raw center of this frame's pattern), and the
    // camera must reach full stereo, which in copy_rescue needs the source's
    // clear convention for the destination's depth direction.
    void check_copies(cadence kind,const char *label){
      // The game has drawn this case since its first frame (run()), so the
      // history reset happened long before effects loaded.
      set_mode(kind);
      warm(label,false);stable(label,60,false);
      require(stereo_observed[drawn-1]>0,"Copy case never produced actual stereo pixels");
      std::printf("PASS %s: the copy's real source is current depth on every Present\n",label);
    }
    // copy_rescue by each route in turn (rescue_route): whatever overwrote the
    // scene's own allocation after the copy, and a full-extent boxed copy,
    // the destination is the captured current depth on every Present.
    static const char *rescue_label(rescue route){
      return route==rescue::rect ? "copy-of-rect-cleared-pass" : route==rescue::pass_clear ? "copy-of-render-pass-cleared-pass" :
        route==rescue::discard ? "copy-of-discarded-pass" : route==rescue::boxed ? "boxed-copy-of-cleared-pass" : "copy-of-cleared-pass";
    }
    void check_rescues(std::initializer_list<rescue> routes){
      for(const auto route:routes){rescue_route=route;stereo_observed={};check_copies(cadence::copy_rescue,rescue_label(route));}
    }
    // The scene draws every other Present (gap_double, gap_ui, gap_overlay) or
    // one Present in 40 (gap_long). In between: a frame presented twice, a
    // colour-only UI pass, or a HUD pass into its own depth-stencil every
    // Present. Once real frames are full stereo, every in-between Present must
    // show its real frame's depth again (tick(): same capture, raw and scene),
    // never flat; with one real frame in 40, the hold ends at the source age.
    void check_gap(cadence kind,const char *label){
      // One real frame in 40 never stays at full strength: warm up drawing
      // every Present first.
      if(kind==cadence::gap_long){continuous(1);}
      set_mode(kind);
      const auto started=clock.now();unsigned consecutive=kind==cadence::gap_long ? 20 : 0;
      while(consecutive<20 && clock.now()-started<15000){tick(label);if(drawn){if(full())++consecutive;else consecutive=0;}}
      require(consecutive>=20,"Real frames between held Presents never reached full stereo");
      if(kind==cadence::gap_long){
        unsigned held_run=0,mono=0,fades=0;std::uint64_t mono_since=0;
        for(unsigned i=0;i<240;++i){
          tick(label);
          if(held)++held_run;else if(!capture_ready){++mono;if(!mono_since)mono_since=step_after;}
          if(drawn){
            // Mono for more than the source age before this real frame.
            if(mono_since && step_before-mono_since>300){require(blend()<1.f,"Depth after a long mono run did not fade in again");++fades;}
            mono_since=0;
          }
        }
        require(held_run>0 && mono>0 && fades>0,"One real frame in 40 did not hold within the source age and then go mono");
        std::printf("PASS %s: %u held, %u mono, %u fade-ins over 240 Presents\n",label,held_run,mono,fades);
        return;
      }
      unsigned real=0,between=0;
      for(unsigned i=0;i<120;++i){
        tick(label);
        if(drawn){require(full(),"A real frame between held Presents lost full stereo");++real;}
        else{require(held && ready() && blend()==1.f,"An in-between Present was not held at full stereo");++between;}
      }
      std::printf("PASS %s: %u real frames full stereo, %u in-between Presents held their real frame's depth\n",label,real,between);
    }
    // The game erases the scene's depth without drawing every other Present
    // (gap_cleared), by ClearDepthStencilView, a render pass whose depth
    // access is CLEAR, one that DISCARDs it, or a clear limited to one
    // full-view or quarter-view rectangle in turn, then on every Present
    // (paused): a screen without its scene, so each such Present holds nothing
    // and is mono from the first (tick()). The real frames stay full stereo,
    // and after the pause the scene recovers.
    void check_gap_cleared(){
      set_mode(cadence::gap_cleared);
      const auto started=clock.now();unsigned consecutive=0;
      while(consecutive<20 && clock.now()-started<15000){tick("gap-cleared");if(drawn){if(full())++consecutive;else consecutive=0;}}
      require(consecutive>=20,"Real frames between cleared Presents never reached full stereo");
      unsigned real=0,cleared=0;
      for(unsigned i=0;i<120;++i){
        tick("gap-cleared");
        if(drawn){require(full(),"A real frame between cleared Presents lost full stereo");++real;}
        else{require(erased && !held && !capture_ready && !ready() && blend()==0.f,"A Present that erased the scene's depth was not mono");++cleared;}
      }
      std::printf("PASS gap-cleared: %u real frames full stereo, %u Presents that cleared or discarded the scene's depth mono\n",real,cleared);
      set_mode(cadence::single,1);warm("before-pause");
      set_mode(cadence::paused);
      const auto paused=clock.now();unsigned mono=0;
      do{tick("pause-erases-depth");++mono;}while(mono<20 || clock.now()-paused<400);
      set_mode(cadence::single,1);warm("after-pause");stable("after-pause",20);
      std::printf("PASS pause: %u Presents that keep erasing the scene's depth mono from the first, then the scene recovers\n",mono);
    }
    // end_discard (draw_end_discard): on every other Present the scene's
    // depth ends the frame DISCARDed by its pass's ending access. Such a
    // Present never shows that frame's depth as current (tick()); it holds the
    // previous Present's depth within the source age, at full stereo (a
    // missing frame). The Presents whose depth a later pass wrote again show
    // their own current depth at full stereo.
    void check_end_discard(){
      set_mode(cadence::end_discard);
      const auto started=clock.now();unsigned consecutive=0;
      while(consecutive<20 && clock.now()-started<15000){
        tick("end-discard");
        if(!end_discarded){if(full())++consecutive;else consecutive=0;}
      }
      require(consecutive>=20,"Presents whose depth a pass wrote again after a DISCARD never reached full stereo");
      unsigned stored=0,discarded=0;
      for(unsigned i=0;i<120;++i){
        tick("end-discard");
        if(!end_discarded){require(full(),"A Present whose depth a pass wrote again after a DISCARD lost its current depth");++stored;}
        else{require(held && ready() && blend()==1.f,"A Present whose scene depth its pass DISCARDed was not held at full stereo");++discarded;}
      }
      std::printf("PASS end-discard: %u Presents whose depth a later pass wrote again full stereo on their own current depth, %u whose pass DISCARDed it held the previous Present's depth\n",stored,discarded);
    }
    // resume_pass (draw_resume_pass): the pass that resumes the scene's pass
    // applies no beginning access, so its CLEAR erases nothing the
    // suspending pass drew: every Present is full stereo on its own current
    // depth, this frame's raw center (pixels(): the GPU's depth, so a runtime
    // that applied the resuming CLEAR fails there), captured over the whole
    // scene (the quarter drawn after the resume does not restart the crop).
    // The list that suspends the pass moves to another method table at its
    // suspending pass (D3D12Core 10.0.26300: a module table its creation never
    // showed) until its next Reset. Until 11-03 the barrier hooks never
    // covered that table, so each of its executions counted as uncovered and
    // unproved every state; the execution hook now discovers it, and once its
    // hooks are installed (during the warm-up) no execution is uncovered.
    void check_resume_pass(){
      set_mode(cadence::resume_pass);
      using uncovered_t=BOOL(*)(std::uint64_t *);
      const auto uncovered_executions=reinterpret_cast<uncovered_t>(GetProcAddress(GetModuleHandleW(L"SunshineSBSTest.addon64"),"SunshineDepthTestUncoveredExecutions"));
      require(uncovered_executions!=nullptr,"The test add-on lacks SunshineDepthTestUncoveredExecutions");
      const auto started=clock.now();unsigned consecutive=0;
      while(consecutive<20 && clock.now()-started<15000){tick("resume-pass");if(full())++consecutive;else consecutive=0;}
      require(consecutive>=20,"A scene drawn in a suspended render pass never reached full stereo");
      std::uint64_t uncovered_before{},uncovered_after{};
      require(uncovered_executions(&uncovered_before),"The uncovered executions were not readable");
      for(unsigned i=0;i<120;++i){
        tick("resume-pass");
        require(full(),"A Present whose scene a suspended render pass drew lost its current depth at the pass that resumed it");
        require(captured.x==0 && captured.y==0 && captured.active_width==scene->width && captured.active_height==scene->height,
          "The CLEAR of the pass that resumed the scene's pass restarted the capture crop at the quarter drawn after it");
      }
      require(uncovered_executions(&uncovered_after),"The uncovered executions were not readable");
      if(uncovered_after!=uncovered_before){
        std::fprintf(stderr,"FAIL resume-pass: %llu of the 120 Presents' executions were uncovered (each unproves every state)\n",
          static_cast<unsigned long long>(uncovered_after-uncovered_before));
        require(false,"An execution of the frame's lists counted as uncovered: a list whose barrier methods no hook covered");
      }
      std::printf("PASS resume-pass: 120 Presents whose scene a suspending render pass drew, resumed by a pass whose CLEAR applies nothing, full stereo on their own current depth; no execution uncovered\n");
    }
    // A new full-view scene starts in another allocation (a cutscene, photo
    // mode or 3D menu) and the old one is left untouched: first a half-size
    // one (C), then one of the same size that was never qualified (B). From
    // the cut Present on nothing is held (it would be the previous camera's
    // depth), so the view is mono until the new allocation's own depth is
    // current; then it recovers to full stereo on it.
    void cut_to(unsigned role,const char *label){
      require(full(),"The Present before a cut did not show full current depth");
      set_mode(cadence::single,role);
      const auto started=clock.now();unsigned mono=0;
      do{
        tick(label);
        require(!held,"A Present after a cut to another allocation held the previous scene's depth");
        // The new allocation has no calibrated camera yet: the cut Present is mono.
        require(mono || (!ready() && blend()==0.f),"The cut Present was not mono");
        if(!ready())++mono;
      }while(!full() && clock.now()-started<15000);
      warm(label);stable(label,20);
      require(stereo_observed[role-1]>0,"The new allocation's scene never produced actual stereo pixels");
      std::printf("PASS %s: %u mono Presents from the cut Present on, nothing held, then full stereo on the new allocation's own depth\n",label,mono);
    }
    void check_gap_cut(){
      continuous(1);
      cut_to(3,"cut-to-half-size-scene");
      cut_to(2,"cut-to-same-size-unqualified-scene");
    }
    void present_second(){
      begin_commands();
      auto *backbuffer=second_buffers[second_chain->GetCurrentBackBufferIndex()].p;
      transition(commands.p,backbuffer,D3D12_RESOURCE_STATE_PRESENT,D3D12_RESOURCE_STATE_COPY_DEST);
      D3D12_TEXTURE_COPY_LOCATION from{},to{};
      from.pResource=source_upload.p;from.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;from.PlacedFootprint=source_footprint;
      to.pResource=backbuffer;to.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      commands->CopyTextureRegion(&to,0,0,0,&from,nullptr);
      transition(commands.p,backbuffer,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_PRESENT);
      checked(commands->Close(),"Close the second window's commands");
      ID3D12CommandList *lists[]{commands.p};
      queue->ExecuteCommandLists(1,lists);
      checked(second_chain->Present(0,0),"The second window's colour-only Present");
      checked(queue->Signal(completion.p,++fence_value),"Fence the second window's Present");
      checked(completion->SetEventOnCompletion(fence_value,completion_event),"Observe the second window's completion");
      require(WaitForSingleObject(completion_event,3000)==WAIT_OBJECT_0,"The second window's Present exceeded three seconds");
      // ReShade's init of the second runtime selects it as observed.runtime.
      if(observed.runtime!=main_runtime){second_runtime=observed.runtime;observed.runtime=main_runtime;}
      sunshine_game3d::test::last_render render{};
      if(!second_runtime || !second_last(second_runtime,&render) || render.sequence<=second_sequence)return;
      second_sequence=render.sequence;++second_renders;
      require(!render.depth.ready && !render.depth.reused_depth && !render.parameters.depth_ready && !render.parameters.camera_ready &&
        render.parameters.strength_blend==0.f,"A second window's colour-only Present showed or held the game window's Generic depth");
    }
    // A second window of the game (its own swapchain on the game's queue, so
    // its own effect runtime) presents colour only after every Present of the
    // game's window, whose scene draws every other Present. The game's window
    // holds its real frame's depth on the Presents between (tick()); the
    // second window's runtime, which never showed that capture, never shows
    // or holds it: the hold belongs to the runtime that showed it.
    // between (--second-swapchain-between): the game's window erases its
    // scene's depth without drawing every other Present (gap_cleared), and on
    // those Presents the second window presents between that work and the
    // game window's Present, so the game window's Present carries no depth
    // work of its own. It still shows a screen without its scene: mono, never
    // its last real frame held (tick()); the second window still shows and
    // holds nothing.
    void check_second_swapchain(bool between=false){
      main_runtime=observed.runtime;
      second_last=reinterpret_cast<decltype(second_last)>(GetProcAddress(GetModuleHandleW(L"SunshineSBSTest.addon64"),"SunshineGame3DTestLastRender"));
      require(main_runtime && second_last,"The second-window case needs the test add-on's last-render query");
      // The trace's first line lists only its own Present's events: the game
      // window's first traced Present comes first (on the game clock the
      // trace opens at the engage, right before this case), so the second
      // runtime's first events, its initial effect reload's among them, are
      // listed (T8) and not taken for setup's. It comes before the second
      // swapchain exists: ReShade's init of the second runtime makes it
      // observed.runtime until present_second() restores the game window's.
      if(!native_trace.has_line())tick(between ? "second-window-between" : "second-window");
      RECT bounds{0,0,LONG(width),LONG(height)};
      require(AdjustWindowRectEx(&bounds,WS_OVERLAPPEDWINDOW,FALSE,WS_EX_NOACTIVATE|WS_EX_TOOLWINDOW),"Could not size the second window");
      second_window=CreateWindowExW(WS_EX_NOACTIVATE|WS_EX_TOOLWINDOW,L"Depth3DRealDepthD3D12Fixture",L"Hidden second game window",WS_OVERLAPPEDWINDOW,
        0,0,bounds.right-bounds.left,bounds.bottom-bounds.top,nullptr,nullptr,GetModuleHandleW(nullptr),nullptr);
      require(second_window && !IsWindowVisible(second_window),"Could not create the hidden second window");
      const auto create_factory=reinterpret_cast<decltype(&CreateDXGIFactory2)>(GetProcAddress(runtime_module,"CreateDXGIFactory2"));
      require(create_factory,"Official runtime lacks its DXGI factory export");
      com_ptr<IDXGIFactory4> factory;
      checked(create_factory(0,IID_PPV_ARGS(factory.put())),"Create the second window's wrapped DXGI factory");
      DXGI_SWAP_CHAIN_DESC1 desc{};
      desc.Width=width;desc.Height=height;desc.Format=source_format;desc.SampleDesc.Count=1;desc.BufferCount=2;
      desc.BufferUsage=DXGI_USAGE_RENDER_TARGET_OUTPUT;desc.SwapEffect=DXGI_SWAP_EFFECT_FLIP_DISCARD;
      com_ptr<IDXGISwapChain1> chain1;
      checked(factory->CreateSwapChainForHwnd(queue.p,second_window,&desc,nullptr,nullptr,chain1.put()),"Create the second window's swapchain");
      checked(chain1->QueryInterface(IID_PPV_ARGS(second_chain.put())),"Query the second window's swapchain");
      checked(second_chain->SetColorSpace1(color==2 ? DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709 : color==3 ? DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020 :
        DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709),"Set the second window's transfer");
      for(unsigned i=0;i<2;++i)checked(second_chain->GetBuffer(i,IID_PPV_ARGS(second_buffers[i].put())),"Get the second window's backbuffer");
      present_second();
      require(second_runtime && second_runtime!=main_runtime,"The second swapchain has no effect runtime of its own");
      second_between=between;
      set_mode(between ? cadence::gap_cleared : cadence::interrupted);
      const char *label=between ? "second-window-between" : "second-window";
      const auto started=clock.now();unsigned consecutive=0;
      while((consecutive<20 || !second_renders) && clock.now()-started<30000){tick(label);if(drawn){if(full())++consecutive;else consecutive=0;}}
      require(consecutive>=20,"Real frames beside a second window never reached full stereo");
      require(second_renders>0,"The second window's runtime never rendered natively");
      const auto renders=second_renders;unsigned real=0,others=0;
      for(unsigned i=0;i<120;++i){
        tick(label);
        if(drawn){require(full(),"A real frame beside a second window lost full stereo");++real;}
        else if(between){require(erased && !held && !capture_ready && !ready() && blend()==0.f,
          "A Present that erased the scene's depth before a second window's Present was not mono");++others;}
        else{require(held && ready() && blend()==1.f,"An in-between Present beside a second window was not held at full stereo");++others;}
      }
      require(second_renders-renders==120,"The second window's runtime did not render on each of its Presents");
      std::printf(between ?
        "PASS second window between: %u real frames full stereo, %u Presents that erased the scene's depth before the second window presented mono; %u colour-only Presents of the second window's own runtime showed and held no depth\n" :
        "PASS second window: %u real frames full stereo, %u in-between Presents held; %u colour-only Presents of the second window's own runtime showed and held no depth\n",
        real,others,second_renders);
    }
    void finish(){
      render_tracked_depth={};
      if(second_chain.p){
        second_chain.reset();
        for(auto &buffer:second_buffers)buffer.reset();
      }
      if(second_window)DestroyWindow(second_window);
      second_window=nullptr;
    }
    void run(const std::string &test_case){
      create_pipeline();normal=false;
      if(test_case=="--crowded-rotating-startup")
        for(unsigned i=0;i<12;++i)unrelated.push_back(target(width,height,1,19));
      // --copy-rescue runs every route in turn; --copy-rescue-<route> one alone,
      // drawn from the game's first frame.
      const bool rescues=test_case.rfind("--copy-rescue",0)==0;
      const bool copies=test_case=="--copy-history" || test_case=="--copy-end" || rescues;
      if(copies){decoy=target(width,height,1,15);scene=target(width,height,1,14);}
      else{scene=target(width,height,1,14);decoy=target(width,height,1,15);}
      // The cut case's first new scene has half the size (another allocation shape).
      third=test_case=="--gap-cut" ? target(width/2,height/2,1,18) : target(width,height,1,18);
      if(test_case=="--copy-history")set_mode(cadence::copy_history);
      if(test_case=="--copy-end")set_mode(cadence::copy_end);
      if(rescues)set_mode(cadence::copy_rescue);
      if(test_case=="--copy-rescue-rect")rescue_route=rescue::rect;
      if(test_case=="--copy-rescue-pass-clear")rescue_route=rescue::pass_clear;
      if(test_case=="--copy-rescue-discard")rescue_route=rescue::discard;
      if(test_case=="--copy-rescue-boxed")rescue_route=rescue::boxed;
      if(test_case=="--end-discard")set_mode(cadence::end_discard);
      if(test_case=="--resume-pass")set_mode(cadence::resume_pass);
      if(test_case.rfind("--gap-",0)==0){
        gap_pipelines();
        if(test_case=="--gap-overlay")overlay=target(width,height,1,19);
        set_mode(test_case=="--gap-double" ? cadence::gap_double : test_case=="--gap-ui" ? cadence::gap_ui :
          test_case=="--gap-overlay" ? cadence::gap_overlay : test_case=="--gap-long" ? cadence::gap_long :
          test_case=="--gap-cleared" ? cadence::gap_cleared : test_case=="--gap-cut" ? cadence::single : cadence::interrupted);
      }
      if(test_case=="--overlap")for(unsigned role=1;role<=3;++role)member(role).pattern=14;
      if(test_case=="--interrupted-startup")set_mode(cadence::interrupted);
      if(test_case=="--moving-startup")set_mode(cadence::moving);
      if(test_case=="--rotating-startup")set_mode(cadence::abc);
      if(test_case=="--crowded-rotating-startup"){
        for(unsigned role=1;role<=3;++role)member(role).draws=8;
        set_mode(cadence::offturn);
      }
      if(test_case=="--flat-startup"){scene->pattern=19;scene->clear_depth=0.f;}
      render_tracked_depth=[&]{draw_depth();};
      trace.open(runtime_directory/"rotation-trajectory.csv");summary.open(runtime_directory/"rotation-summary.csv");
      trace<<std::setprecision(17)<<"phase,game_ms,render,drawn,drawn_original,current_original,lifetime,layout,frame,capture_ready,camera_ready,H,t0,blend,marker_half,eye_max_difference,raw_center,crop_x,crop_y,crop_width,crop_height,raster_mask,held\n";
      summary<<"phase,frames,A,B,C,result\n";
      const auto until=clock.wall_now()+45000;
      while((!observed.runtime || !observed.renders) && clock.wall_now()<until)step();
      require(observed.runtime && observed.renders && !observed.inject,"Actual rotation fixture did not initialize");
      check_unified_addon();
      game3d.start();game3d.attach_export();game3d.set_strength(100);
      rendered_sequence=game3d.await_render([&]{update_marker();step();});
      // Setup is done: game time and the trace from here on. The cadence
      // (which member draws, which Presents have no depth) and a moving
      // scene go on, from their first step again: setup's Presents, whose
      // number is real time, do not choose where game time starts in them.
      // --wall-clock keeps the old trace (from start()) and cadence.
      game3d.begin_game_time();
      if(clock.engaged()){phase_present=0;movement_started=0;}
      for(unsigned band=0;band<4;++band)for(unsigned i=0;i<8;++i)points[band*8+i]={(2*band+1)*width/8+i-4,height/4};
      buffer(pixel_readback,UINT64(slots)*512,D3D12_HEAP_TYPE_READBACK);
      if(!test_case.empty()){
        if(test_case=="--overlap")check_overlap();
        else if(test_case=="--interrupted-startup")check_interrupted_startup();
        else if(test_case=="--moving-startup")check_moving_startup();
        else if(test_case=="--rotating-startup")check_rotating_startup();
        else if(test_case=="--crowded-rotating-startup")check_crowded_startup();
        else if(test_case=="--flat-startup")check_flat_startup();
        else if(test_case=="--layout")check_layout();
        else if(test_case=="--copy-history")check_copies(cadence::copy_history,"copy-history-before-draw");
        else if(test_case=="--copy-end")check_copies(cadence::copy_end,"copy-after-draw");
        else if(test_case=="--copy-rescue")check_rescues({rescue::clear,rescue::rect,rescue::pass_clear,rescue::discard,rescue::boxed});
        else if(rescues)check_rescues({rescue_route});
        else if(test_case=="--gap-none")check_gap(cadence::interrupted,"gap-no-depth-work");
        else if(test_case=="--gap-double")check_gap(cadence::gap_double,"gap-double-present");
        else if(test_case=="--gap-ui")check_gap(cadence::gap_ui,"gap-color-only-ui");
        else if(test_case=="--gap-overlay")check_gap(cadence::gap_overlay,"gap-own-hud-depth");
        else if(test_case=="--gap-long")check_gap(cadence::gap_long,"gap-one-real-frame-in-40");
        else if(test_case=="--gap-cleared")check_gap_cleared();
        else if(test_case=="--gap-cut")check_gap_cut();
        else if(test_case=="--second-swapchain")check_second_swapchain();
        else if(test_case=="--second-swapchain-between")check_second_swapchain(true);
        else if(test_case=="--end-discard")check_end_discard();
        else if(test_case=="--resume-pass")check_resume_pass();
        else throw std::runtime_error("Unknown isolated rotation case");
        finish();return;
      }
      continuous(1);continuous(2);continuous(3);
      set_mode(cadence::abc);warm("ABC-independent-warmup");stable("ABC-independent",90);
      set_mode(cadence::aabb);warm("AABB-independent-warmup");stable("AABB-independent",64);
      set_mode(cadence::offturn);warm("offturn-transfer-warmup");stable("offturn-transfer",60);
      set_mode(cadence::offturn_history);warm("offturn-history-warmup");stable("offturn-history",60);

      const auto old_b=lifetimes[1];decoy.reset();decoy=target(width,height,1,14);set_mode(cadence::abc);
      warm("member-B-replacement",true,[&](std::uint64_t elapsed){
        // A held Present shows another member's previous capture with its own camera.
        if(drawn==2 && elapsed<650 && !held)require(!ready(),"New resource inherited old initialized camera");
        if(current(2))require(captured.source_id!=old_b,"Replacement reused destroyed lifetime identity");
      });
      stable("replacement-independent-basis",30);

      const auto old_crop=crops[0];const auto crop_lifetime=lifetimes[0];
      require(old_crop==std::array<std::uint32_t,4>{0,0,width,height},"Crop-change control did not start from a captured full view");
      scene->partial_viewport=true;scene->pattern=15;set_mode(cadence::single,1);
      const auto crop_until=clock.now()+12000;unsigned crop_frames=0;
      do{
        tick("member-A-unsupported-crop");
        require(!ready(),"Unsupported partial crop reused a ready full-frame camera");
        if(current(1)){
          // The end-of-frame copy's crop is the frame's largest viewport, now
          // the scene's quarter. The controller key includes the exact crop
          // independently; require that real change on the same live
          // allocation, not a fabricated epoch increment.
          require(captured.source_id==crop_lifetime && crops[0]!=old_crop &&
                  crops[0]==std::array<std::uint32_t,4>{0,0,width/2,height/2},
                  "Partial-viewport control did not publish the changed copy crop on its original live resource");
          ++crop_frames;
        }
      }while(crop_frames<16 && clock.now()<crop_until);
      require(crop_frames>=16,"Unsupported-crop control never provided actual current captured depth");
      scene->partial_viewport=false;scene->pattern=14;set_mode(cadence::single,1);
      warm("full-layout-fresh-reference",true,[&](std::uint64_t elapsed){
        if(elapsed<650)require(!ready(),"Restored full layout reused an old numeric reference");
      });
      set_mode(cadence::abc);warm("full-layout-return");stable("full-layout-return",30);
      color_only_gap("true-color-only-gap");
      require_fade_in(cadence::abc,1,"gap-current-depth-return");
      set_mode(cadence::abc);warm("gap-current-depth-return");stable("gap-current-depth-return",30);

      for(unsigned r=1;r<=3;++r)member(r).pattern=14;
      require(game3d.recalibrate(),"Cannot establish fresh equal-basis controls");set_mode(cadence::abc);
      warm("same-basis-warmup");stable("same-basis-steady",30);
      drift=true;set_mode(cadence::abc);stable("same-basis-slow-center",120,false);drift=false;
      // H/t0 and actual eye-difference traces support continuity review; no tiny
      // post-hoc threshold is invented for independently timed controllers.
      for(unsigned r=0;r<3;++r)require(stereo_observed[r]>0,"A member never produced nonzero actual stereo separation");
      std::puts("PASS actual rotating sources: ABC/AABB, independent raw bases, current-color SBS, off-turn and history copies into live members, lifetime/layout reset, gaps, same-basis trajectory");
      finish();
    }
  };
}

int main(int argc,char **argv){
  std::setvbuf(stdout,nullptr,_IONBF,0);
  sunshine_game3d_fixture::take_wall_clock_option(argc,argv);
  if(argc!=5 && argc!=6){std::fputs("usage: depth_alternating_runtime <official.dll> <frozenShaders> <test.addon64> <fresh-output> [--overlap|--interrupted-startup|--moving-startup|--rotating-startup|--crowded-rotating-startup|--flat-startup|--layout|--copy-history|--copy-end|--copy-rescue|--copy-rescue-rect|--copy-rescue-pass-clear|--copy-rescue-discard|--copy-rescue-boxed|--gap-none|--gap-double|--gap-ui|--gap-overlay|--gap-long|--gap-cleared|--gap-cut|--second-swapchain|--second-swapchain-between|--end-discard|--resume-pass] [--wall-clock]\n",stderr);return 2;}
  std::thread([]{Sleep(360000);std::fputs("FAIL rotation watchdog\n",stderr);TerminateProcess(GetCurrentProcess(),124);}).detach();
  try{
    const std::string test_case=argc==6 ? argv[5] : "";
    require(test_case.empty() || test_case=="--overlap" || test_case=="--interrupted-startup" || test_case=="--moving-startup" || test_case=="--rotating-startup" || test_case=="--crowded-rotating-startup" || test_case=="--flat-startup" || test_case=="--layout" ||
      test_case=="--copy-history" || test_case=="--copy-end" || test_case=="--copy-rescue" || test_case=="--copy-rescue-rect" ||
      test_case=="--copy-rescue-pass-clear" || test_case=="--copy-rescue-discard" || test_case=="--copy-rescue-boxed" ||
      test_case=="--gap-none" || test_case=="--gap-double" ||
      test_case=="--gap-ui" || test_case=="--gap-overlay" || test_case=="--gap-long" || test_case=="--gap-cleared" || test_case=="--gap-cut" ||
      test_case=="--second-swapchain" || test_case=="--second-swapchain-between" || test_case=="--end-discard" ||
      test_case=="--resume-pass","Unknown isolated rotation case");
    select_native_boot();
    width=3840;height=2160;require(!fs::exists(fs::absolute(argv[4])),"Fresh rotation output required");
    rotation_fixture f;f.runtime_directory=fs::absolute(argv[4]);
    f.initialize(fs::absolute(argv[1]),fs::absolute(argv[2]),f.runtime_directory,2,0,fs::absolute(argv[3]));f.run(test_case);f.clock.report("alternating");return 0;
  }catch(const std::exception &e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
