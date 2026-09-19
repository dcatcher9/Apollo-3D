// SPDX-License-Identifier: GPL-3.0-only
// Actual rotation regression: unchanged executable/assertions for old/new add-ons.
#include "test_raw_runtime_fixture.h"
#include "depth_addon.h"

namespace {
  using query_t=BOOL (*)(api::effect_runtime *,sunshine_depth::frame_depth *);
  query_t query=nullptr;
  sunshine_depth::frame_depth captured;
  unsigned captured_render=0;
  bool capture_ready=false;
  void observe_rotation(api::effect_runtime *runtime,api::effect_technique technique,
      api::command_list *,api::resource_view,api::resource_view) {
    char name[256]{};runtime->get_technique_name(technique,name);
    if(!named(name,technique_name))return;
    captured={};capture_ready=query && query(runtime,&captured) && captured.ready;
    captured_render=observed.renders;
  }

  struct rotation_fixture:raw_runtime_fixture {
    enum class cadence { single,abc,aabb,offturn,missing,concurrent,interrupted,moving,auxiliary };
    using action_t=BOOL (*)(api::effect_runtime *);
    using select_t=BOOL (*)(api::effect_runtime *,std::uint64_t);
    using state_t=BOOL (*)(api::effect_runtime *,std::uint64_t *,BOOL *);
    action_t recalibrate=nullptr;select_t select_manual=nullptr;state_t manual_state=nullptr;
    std::unique_ptr<target_t> third;
    std::vector<std::unique_ptr<target_t>> unrelated;
    cadence mode=cadence::single;
    unsigned single_role=1,phase_present=0,drawn=0,marker_sequence=0;
    unsigned raster_mask=0;
    std::uint64_t movement_started=0;
    bool drift=false;
    std::uint16_t marker=0x3400;
    std::array<std::uint64_t,3> lifetimes{},layouts{};
    std::array<std::array<std::uint32_t,4>,3> crops{};
    std::array<unsigned,3> stereo_observed{};
    std::ofstream trace,summary;
    com_ptr<ID3D12Resource> pixel_readback;
    struct point {unsigned x,y;};
    std::array<point,32> points{};
    static constexpr unsigned count=32,slots=count*3+1;
    float eye_difference=0;

    target_t &member(unsigned role){return role==1 ? *scene : role==2 ? *decoy : *third;}
    static float center(unsigned pattern){
      if(pattern==18)return .0625f;
      if(pattern==19)return .5f;
      if(pattern>=100 && pattern<=228)return .125f+float(pattern-100)/1024.f;
      return pattern==15 ? .25f : .125f;
    }
    void set_mode(cadence next,unsigned role=1){mode=next;single_role=role;phase_present=0;}
    void draw_auxiliary_scene(){
      // A non-scene pass uses the same allocation with a changing valid larger
      // viewport. Its later clear is followed by the actual half-view scene;
      // the final legal unbind copy, rather than total-frame workload, owns
      // the captured crop and layout.
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
      commands->ClearDepthStencilView(dsv,D3D12_CLEAR_FLAG_DEPTH,0.f,0,0,nullptr);
      const D3D12_VIEWPORT captured_view{0,0,float(width/2),float(height/2),0,1};
      const D3D12_RECT captured_scissor{0,0,LONG(width/2),LONG(height/2)};
      commands->RSSetViewports(1,&captured_view);commands->RSSetScissorRects(1,&captured_scissor);
      constants={captured_view.Width,captured_view.Height,target.pattern};
      commands->SetGraphicsRoot32BitConstants(0,3,&constants,0);
      for(unsigned i=0;i<4;++i)commands->DrawInstanced(3,1,0,0);
      commands->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    }
    void draw_depth(){
      drawn=0;raster_mask=0;
      if(mode==cadence::single)drawn=single_role;
      else if(mode==cadence::abc || mode==cadence::offturn)drawn=phase_present%3+1;
      else if(mode==cadence::aabb)drawn=(phase_present/2)%2+1;
      else if(mode==cadence::concurrent || mode==cadence::moving || mode==cadence::auxiliary)drawn=1;
      else if(mode==cadence::interrupted)drawn=phase_present%2 ? 0 : 1;
      if(mode==cadence::moving){
        if(!movement_started)movement_started=GetTickCount64();
        // Looking repeatedly between coherent near/middle/far center patches
        // begins before the first effect frame. No stationary calibration lead-in.
        constexpr unsigned patterns[]{18,14,15};
        scene->pattern=patterns[((GetTickCount64()-movement_started)/300)%3];
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
      const auto draw_unrelated=[&]{
        for(unsigned i=0;i<unrelated.size();++i)
          draw(*unrelated[(i+phase_present)%unrelated.size()]);
      };
      if(!unrelated.empty() && phase_present%2)draw_unrelated();
      auto &current=member(drawn);draw(current);
      if(mode==cadence::offturn){
        // Real write activity on another live member, without a subsequent DSV
        // clear/unbind authorizing preserve2 to capture that destination.
        auto &other=member(drawn%3+1);
        transition(commands.p,current.texture.p,D3D12_RESOURCE_STATE_DEPTH_WRITE,D3D12_RESOURCE_STATE_COPY_SOURCE);
        transition(commands.p,other.texture.p,D3D12_RESOURCE_STATE_DEPTH_WRITE,D3D12_RESOURCE_STATE_COPY_DEST);
        commands->CopyResource(other.texture.p,current.texture.p);
        transition(commands.p,other.texture.p,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_DEPTH_WRITE);
        transition(commands.p,current.texture.p,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_DEPTH_WRITE);
      }
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
      return capture_ready && captured_render==observed.renders &&
        captured.source_resource.handle==reinterpret_cast<std::uint64_t>(m.texture.p) &&
        captured.width==m.width && captured.height==m.height;
    }
    bool full(){return drawn && current(drawn) && ready() && scalar("Sunshine_CameraStrengthBlend")==1.f;}
    void require_basis(){
      require(drawn && current(drawn),"No actual current member for basis assertion");
      const float t=center(member(drawn).pattern);
      require(scalar("Sunshine_CameraDepthScale")==1.f/t && zero()[1]==t,
        "Physical member inherited another member's H/t0 instead of its own exact raw basis");
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
      // Compact actual native readbacks each present, no fitted registration.
      begin_commands();
      transition(commands.p,exported.p,D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE);
      transition(commands.p,mono.p,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COPY_SOURCE);
      for(unsigned i=0;i<count;++i){
        copy_pixel(exported.p,points[i].x,points[i].y,i*3,DXGI_FORMAT_R16G16B16A16_FLOAT);
        copy_pixel(exported.p,width+points[i].x,points[i].y,i*3+1,DXGI_FORMAT_R16G16B16A16_FLOAT);
        copy_pixel(mono.p,points[i].x,points[i].y,i*3+2,DXGI_FORMAT_R16G16B16A16_FLOAT);
      }
      transition(commands.p,mono.p,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_COPY_DEST);
      transition(commands.p,exported.p,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
      // The flat-startup oracle also reads the original game allocation when
      // the selector correctly rejects its copy. A configured draw alone is
      // not evidence that the real depth image contains the intended .5.
      const bool direct_flat=!capture_ready && drawn && member(drawn).pattern==19;
      auto *depth_resource=capture_ready ? reinterpret_cast<ID3D12Resource *>(captured.resource.handle) :
        direct_flat ? member(drawn).texture.p : nullptr;
      if(depth_resource){
        const auto before=direct_flat ? D3D12_RESOURCE_STATE_DEPTH_WRITE : D3D12_RESOURCE_STATE_COPY_DEST;
        transition(commands.p,depth_resource,before,D3D12_RESOURCE_STATE_COPY_SOURCE);
        copy_pixel(depth_resource,direct_flat ? member(drawn).width/2 : captured.x+captured.active_width/2,
          direct_flat ? member(drawn).height/2 : captured.y+captured.active_height/2,slots-1,DXGI_FORMAT_R32_TYPELESS);
        transition(commands.p,depth_resource,D3D12_RESOURCE_STATE_COPY_SOURCE,before);
      }
      submit();
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
          const float left=value(i*3,c),right=value(i*3+1,c),native=value(i*3+2,c);
          require(std::isfinite(left)&&std::isfinite(right)&&std::isfinite(native),"Current native stereo/mono has nonfinite RGBA");
          require(native==expected,"Physical mono is stale or differs from the current source");
          if(c==1)require(std::abs(left-expected)<.001f && std::abs(right-expected)<.001f,"SBS carries a stale/wrong current-frame color marker");
          if(!ready())require(std::abs(left-expected)<.003f && std::abs(right-expected)<.003f,"Unavailable depth did not show actual current mono color");
          if(c==0 || c==2)eye_difference=std::max(eye_difference,std::abs(left-right));
        }
      }
      if(depth_resource){
        float raw=0;std::memcpy(&raw,bytes+size_t(slots-1)*512,4);
        require(drawn && raw==center(member(drawn).pattern),"Current backup contains stale/wrong-member raw center depth");
      }
      const D3D12_RANGE no_write{0,0};pixel_readback->Unmap(0,&no_write);
      if(drawn && full() && eye_difference>.002f)++stereo_observed[drawn-1];
    }
    void tick(const char *phase){
      update_marker();step();
      require(captured_render==observed.renders,"Missing actual technique observation");
      require(!capture_ready || (drawn && current(drawn)),"Wrong/old physical depth admitted for this rendered present");
      require(!ready() || capture_ready,"Camera ready without current legally preserved depth");
      require(drawn || (!capture_ready && !ready() && scalar("Sunshine_CameraStrengthBlend")==0.f),"True color-only gap reused prior depth or stereo");
      if(capture_ready){
        lifetimes[drawn-1]=captured.source_id;layouts[drawn-1]=captured.layout_epoch;
        crops[drawn-1]={captured.x,captured.y,captured.active_width,captured.active_height};
      }
      pixels();
      trace<<phase<<','<<GetTickCount64()<<','<<observed.renders<<','<<drawn<<','
        <<(drawn?reinterpret_cast<std::uint64_t>(member(drawn).texture.p):0)<<','<<captured.source_resource.handle<<','
        <<captured.source_id<<','<<captured.layout_epoch<<','<<captured.frame_index<<','<<capture_ready<<','<<ready()<<','
        <<scalar("Sunshine_CameraDepthScale")<<','<<zero()[1]<<','<<scalar("Sunshine_CameraStrengthBlend")<<','
        <<marker<<','<<eye_difference<<','<<(drawn?center(member(drawn).pattern):0.f)<<','
        <<captured.x<<','<<captured.y<<','<<captured.active_width<<','<<captured.active_height<<','<<raster_mask<<'\n';
      require(trace.good(),"Cannot record actual-frame rotation trajectory");
    }
    void warm(const char *phase,bool basis=true,const std::function<void(std::uint64_t)> &guard={}){
      const auto started=GetTickCount64();unsigned consecutive=0;
      do{
        tick(phase);if(guard)guard(GetTickCount64()-started);
        if(full()){if(basis)require_basis();++consecutive;}else consecutive=0;
      }while(consecutive<12 && GetTickCount64()-started<15000);
      std::printf("MEASURE %s warmup_ms=%llu consecutive_full=%u\n",phase,static_cast<unsigned long long>(GetTickCount64()-started),consecutive);
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
    void missing_then_recover(){
      set_mode(cadence::missing);for(unsigned i=0;i<8;++i)tick("true-color-only-gap");
      set_mode(cadence::single,1);warm("current-depth-return");stable("current-depth-return",20);
    }
    void check_overlap(){
      continuous(1); // No manual selection: A qualifies before equal-quality peers.
      set_mode(cadence::abc);warm("overlap-ABC-control");stable("overlap-ABC-control",30);
      set_mode(cadence::concurrent);
      // Persistent overlap must not lock an already-qualified family into mono.
      // The expected winner A was specified before any concurrent result is read.
      warm("concurrent-anchor-recovery");stable("concurrent-anchor",60);
      set_mode(cadence::abc);warm("concurrent-back-to-ABC");stable("concurrent-back-to-ABC",30);
      missing_then_recover();
      require(stereo_observed[0]>0,"Concurrent preferred source never produced actual stereo pixels");
      std::puts("PASS overlap: qualified A/ABC, persistent concurrent raster with current A, return to ABC, current mono on real gaps");
    }
    void check_interrupted_startup(){
      const auto started=GetTickCount64();unsigned valid=0,missing=0,consecutive_valid=0;
      do{
        tick("interrupted-startup");
        if(drawn){
          ++valid;
          if(full()){require_basis();++consecutive_valid;}else consecutive_valid=0;
        }else ++missing; // tick() already verifies exact current mono/zero blend.
      }while(consecutive_valid<30 && GetTickCount64()-started<15000);
      std::printf("MEASURE interrupted-startup elapsed_ms=%llu valid=%u missing=%u full_valid_run=%u\n",
        static_cast<unsigned long long>(GetTickCount64()-started),valid,missing,consecutive_valid);
      require(missing>=30 && consecutive_valid==30,
        "Interleaved missing presents prevented async exact-source startup/full-strength on valid frames");
      require(stereo_observed[0]>0,"Recovered valid frames never produced actual stereo pixels");
      missing_then_recover();
      std::puts("PASS interrupted startup: actual async samples survive missing presents; valid frames full stereo, absent frames current mono");
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
      std::puts("PASS rotating startup: initial ABC without prior history, exact H8/H4/H16 bases,90 current-depth/current-color/full-stereo frames");
    }
    void check_crowded_startup(){
      // Every allocation is new. Older-lifetime unrelated buffers render on
      // every present, so a serial lifetime-ID tour cannot hide behind prior
      // source qualification. Content/camera state is never injected.
      const auto started=GetTickCount64();unsigned consecutive=0;
      do{
        tick("crowded-fresh-ABC-startup");
        if(full()){require_basis();++consecutive;}else consecutive=0;
      }while(consecutive<12 && GetTickCount64()-started<8000);
      std::printf("MEASURE crowded-startup warmup_ms=%llu consecutive_full=%u unrelated=%zu\n",
        static_cast<unsigned long long>(GetTickCount64()-started),consecutive,unrelated.size());
      require(consecutive==12,"Fresh rotating sources among12 unrelated buffers did not reach full stereo within8s");
      stereo_observed={};stable("crowded-ABC-sustained",90);
      for(unsigned role=0;role<3;++role){
        require(lifetimes[role]!=0 && stereo_observed[role]>0,
          "Crowded rotating member never supplied its own current depth and actual stereo pixels");
        for(unsigned other=0;other<role;++other)
          require(lifetimes[role]!=lifetimes[other],"Crowded rotating resources share a lifetime identity");
      }
      std::puts("PASS crowded startup:12 older flat active buffers, fresh ABC H8/H4/H16, transfer-only offturn writes,90 current-color/current-depth/full-stereo frames");
    }
    void check_flat_startup(){
      // A has rendered the same .5 plane since the first game draw, before
      // effects loaded. It is written depth with a known clear convention,
      // but cannot supply a useful scene reference. No UI actions are used.
      const auto original=scene->texture.p;
      const auto initial_started=GetTickCount64();unsigned flat_frames=0;
      do{
        tick("flat-startup-mono");++flat_frames;
        require(!ready() && scalar("Sunshine_CameraStrengthBlend")==0.f,
          "Flat interior depth initialized a camera instead of remaining current mono");
      }while(GetTickCount64()-initial_started<2600);
      std::printf("PASS initial flat depth: %u actual .5 draws over%llu ms, current mono throughout\n",flat_frames,
        static_cast<unsigned long long>(GetTickCount64()-initial_started));
      scene->pattern=14;
      warm("flat-to-useful-fresh-reference");stable("flat-to-useful-sustained",30);
      const auto useful_lifetime=lifetimes[0];
      require(scene->texture.p==original && useful_lifetime && stereo_observed[0]>0,
        "Flat-to-useful phase did not reach actual stereo on its original live allocation");
      // A previously valid target may remain usable until its1500 ms expiry.
      // Rejected flat samples must never move H/t0, even during that interval.
      scene->pattern=19;
      const auto flat_return_started=GetTickCount64();unsigned expired_mono=0;
      do{
        tick("useful-to-flat-hold");
        require(scalar("Sunshine_CameraDepthScale")==8.f && zero()[1]==.125f,
          "Rejected flat samples changed an established useful source's H/t0");
        if(current(1))require(captured.source_id==useful_lifetime,"Same live flat source changed lifetime identity");
        if(GetTickCount64()-flat_return_started>=1500){
          require(!ready() && scalar("Sunshine_CameraStrengthBlend")==0.f,
            "Flat-only evidence kept stale ready depth beyond target expiry");
          ++expired_mono;
        }
      }while(GetTickCount64()-flat_return_started<2600);
      require(expired_mono>0,"Returning-flat phase did not exercise expired-evidence current mono");
      scene->pattern=14;stereo_observed[0]=0;
      warm("flat-return-useful-retained-reference");stable("flat-return-useful-sustained",30);
      require(scene->texture.p==original && lifetimes[0]==useful_lifetime && stereo_observed[0]>0,
        "Useful return changed lifetime or never restored actual current stereo");
      std::puts("PASS flat content admission: no H2 startup, fresh H8/t0.125 on useful depth, held basis through later flat rejection and same-source stereo recovery");
    }
    void check_moving_startup(){
      const auto started=GetTickCount64();unsigned consecutive=0,changes=0,previous=scene->pattern;
      do{
        tick("moving-center-startup");
        if(scene->pattern!=previous){++changes;previous=scene->pattern;}
        if(full()){
          const float H=scalar("Sunshine_CameraDepthScale"),t0=zero()[1];
          require(std::isfinite(H) && H>=4.f && H<=16.f && std::isfinite(t0) && t0>=.0625f && t0<=.25f,
            "Moving-source camera escaped the independently known raw range");
          ++consecutive;
        }else consecutive=0;
      }while((consecutive<60 || changes<6) && GetTickCount64()-started<15000);
      std::printf("MEASURE moving-center-startup elapsed_ms=%llu center_changes=%u full_run=%u\n",
        static_cast<unsigned long long>(GetTickCount64()-started),changes,consecutive);
      require(consecutive>=60 && changes>=6,"Continuous valid moving depth never reached persistent full stereo within15s");
      require(stereo_observed[0]>0,"Moving-source initialization never reached actual stereo pixels");
      set_mode(cadence::missing);for(unsigned i=0;i<8;++i)tick("moving-real-gap-mono");
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
      const auto until=GetTickCount64()+10000;
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
      }while(good<30 && GetTickCount64()<until);
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
    void finish(){
      observed.capture=false;render_tracked_depth={};reshade::unregister_event<reshade::addon_event::reshade_render_technique>(observe_rotation);
    }
    void run(const std::string &test_case){
      create_pipeline();normal=false;
      if(test_case=="--crowded-rotating-startup")
        for(unsigned i=0;i<12;++i)unrelated.push_back(target(width,height,1,19));
      scene=target(width,height,1,14);decoy=target(width,height,1,15);third=target(width,height,1,18);
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
      trace<<std::setprecision(17)<<"phase,wall_ms,render,drawn,drawn_original,current_original,lifetime,layout,frame,capture_ready,camera_ready,H,t0,blend,marker_half,eye_max_difference,raw_center,crop_x,crop_y,crop_width,crop_height,raster_mask\n";
      summary<<"phase,frames,A,B,C,result\n";
      const auto until=GetTickCount64()+45000;
      while((!observed.runtime || !observed.renders) && GetTickCount64()<until)step();
      require(observed.runtime && observed.renders && !observed.inject,"Actual rotation fixture did not initialize");
      check_unified_addon();
      const auto module=GetModuleHandleW(L"SunshineSBSTest.addon64");
      query=reinterpret_cast<query_t>(GetProcAddress(module,"SunshineDepthTestFrame"));
      recalibrate=reinterpret_cast<action_t>(GetProcAddress(module,"SunshineGame3DTestRecalibrate"));
      select_manual=reinterpret_cast<select_t>(GetProcAddress(module,"SunshineDepthTestSelectManual"));
      manual_state=reinterpret_cast<state_t>(GetProcAddress(module,"SunshineDepthTestManualState"));
      require(query && recalibrate && select_manual && manual_state,"Existing passive-frame/UI-action adapters required");
      reshade::register_event<reshade::addon_event::reshade_render_technique>(observe_rotation);
      set_int("Depth_Map_View",0);set_float("Depth_Adjustment",100);set_float("Sharpen_Power",0);
      observed.runtime->set_uniform_value_bool(uniform("Show_Infill_Mask"),false);
      find_texture("DoubleTex",exported,width*2,DXGI_FORMAT_R16G16B16A16_FLOAT);
      observed.capture=true; // Existing hook copies THIS present's physical mono.
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
        else throw std::runtime_error("Unknown isolated rotation case");
        finish();return;
      }
      continuous(1);continuous(2);continuous(3);
      set_mode(cadence::abc);warm("ABC-independent-warmup");stable("ABC-independent",90);
      set_mode(cadence::aabb);warm("AABB-independent-warmup");stable("AABB-independent",64);
      set_mode(cadence::offturn);warm("offturn-transfer-warmup");stable("offturn-transfer",60);

      const auto pinned=reinterpret_cast<std::uint64_t>(scene->texture.p);
      require(select_manual(observed.runtime,pinned),"Cannot pin real A");continuous(1);
      set_mode(cadence::abc);
      for(unsigned i=0;i<36;++i){
        tick("manual-pin-ABC");std::uint64_t actual=0;BOOL recovering=TRUE;
        require(manual_state(observed.runtime,&actual,&recovering) && actual==pinned && !recovering,"Live manual pin was replaced or entered recovery");
        if(drawn!=1)require(!capture_ready && !ready(),"Manual pin silently switched to another rotating member");
      }
      require(select_manual(observed.runtime,0),"Cannot release real manual pin");set_mode(cadence::abc);
      warm("manual-release-warmup");stable("manual-release",30);

      const auto old_b=lifetimes[1];decoy.reset();decoy=target(width,height,1,14);set_mode(cadence::abc);
      warm("member-B-replacement",true,[&](std::uint64_t elapsed){
        if(drawn==2 && elapsed<650)require(!ready(),"New resource inherited old initialized camera");
        if(current(2))require(captured.source_id!=old_b,"Replacement reused destroyed lifetime identity");
      });
      stable("replacement-independent-basis",30);

      const auto old_crop=crops[0];const auto crop_lifetime=lifetimes[0];
      require(old_crop==std::array<std::uint32_t,4>{0,0,width,height},"Crop-change control did not start from a captured full view");
      scene->segmented_viewport=true;scene->pattern=15;set_mode(cadence::single,1);
      const auto crop_until=GetTickCount64()+12000;unsigned crop_frames=0;
      do{
        tick("member-A-unsupported-crop");
        require(!ready(),"Unsupported partial crop reused a ready full-frame camera");
        if(current(1)){
          // The total-draw viewport epoch can remain unchanged while the
          // preserved copy's crop changes. The controller key includes the
          // exact crop independently; require that real change on the same
          // live allocation, not a fabricated epoch increment.
          require(captured.source_id==crop_lifetime && crops[0]!=old_crop &&
                  crops[0]==std::array<std::uint32_t,4>{0,0,width/2,height/2},
                  "Segmented control did not publish the changed copy crop on its original live resource");
          ++crop_frames;
        }
      }while(crop_frames<16 && GetTickCount64()<crop_until);
      require(crop_frames>=16,"Unsupported-crop control never provided actual current captured depth");
      scene->segmented_viewport=false;scene->pattern=14;set_mode(cadence::single,1);
      warm("full-layout-fresh-reference",true,[&](std::uint64_t elapsed){
        if(elapsed<650)require(!ready(),"Restored full layout reused an old numeric reference");
      });
      set_mode(cadence::abc);warm("full-layout-return");stable("full-layout-return",30);
      set_mode(cadence::missing);for(unsigned i=0;i<8;++i)tick("true-color-only-gap");
      set_mode(cadence::abc);warm("gap-current-depth-return");stable("gap-current-depth-return",30);

      for(unsigned r=1;r<=3;++r)member(r).pattern=14;
      require(recalibrate(observed.runtime),"Cannot establish fresh equal-basis controls");set_mode(cadence::abc);
      warm("same-basis-warmup");stable("same-basis-steady",30);
      drift=true;set_mode(cadence::abc);stable("same-basis-slow-center",120,false);drift=false;
      // H/t0 and actual eye-difference traces support continuity review; no tiny
      // post-hoc threshold is invented for independently timed controllers.
      for(unsigned r=0;r<3;++r)require(stereo_observed[r]>0,"A member never produced nonzero actual stereo separation");
      std::puts("PASS actual rotating sources: ABC/AABB, independent raw bases, current-color SBS, transfer-only decoys, pin/release, lifetime/layout reset, gaps, same-basis trajectory");
      finish();
    }
  };
}

int main(int argc,char **argv){
  std::setvbuf(stdout,nullptr,_IONBF,0);
  if(argc!=5 && argc!=6){std::fputs("usage: depth_alternating_runtime <official.dll> <Shaders> <test.addon64> <fresh-output> [--overlap|--interrupted-startup|--moving-startup|--rotating-startup|--crowded-rotating-startup|--flat-startup|--layout]\n",stderr);return 2;}
  std::thread([]{Sleep(360000);std::fputs("FAIL rotation watchdog\n",stderr);TerminateProcess(GetCurrentProcess(),124);}).detach();
  try{
    const std::string test_case=argc==6 ? argv[5] : "";
    require(test_case.empty() || test_case=="--overlap" || test_case=="--interrupted-startup" || test_case=="--moving-startup" || test_case=="--rotating-startup" || test_case=="--crowded-rotating-startup" || test_case=="--flat-startup" || test_case=="--layout","Unknown isolated rotation case");
    require(sunshine_camera_fixture::flag("SUNSHINE_DEPTH_BIND_SWITCH_TEST") && sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC") &&
      sunshine_camera_fixture::flag("SUNSHINE_GAME3D_AUTOMATIC_ACTIONS_TEST"),"Explicit preserve2 and Automatic test-action flags required");
    width=3840;height=2160;require(!fs::exists(fs::absolute(argv[4])),"Fresh rotation output required");
    rotation_fixture f;f.runtime_directory=fs::absolute(argv[4]);
    f.initialize(fs::absolute(argv[1]),fs::absolute(argv[2]),f.runtime_directory,2,0,fs::absolute(argv[3]));f.run(test_case);return 0;
  }catch(const std::exception &e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
