// Includes actual producer observations and an explicitly authored owner-only record seam.
    void foreground_edge_enabled(bool enabled) {
      program_.foreground_edge_shaders[0u] = enabled ? foreground_edge_shaders_[0u].Get() : nullptr;
    }

    struct foreground_observation_t {
      std::vector<UINT> mask, data, cbc, raw;
      std::map<std::string, std::vector<char>> immutable;
    };

    foreground_observation_t foreground_observation() {
      foreground_observation_t o;
      o.mask=words(outline_matte());o.data=calibration();o.cbc=receipt();o.raw=ring_receipt();
      const auto e=helper_.enclosure_views();const auto f=helper_.mutable_views();
      o.immutable={{"RGB",read(rgb_.resource.Get())},{"parentRGB",read(origin_.resource.Get())},
        {"P",read(probability_.resource.Get())},{"S",read(support_.resource.Get())},{"OCR",read(ocr_.resource.Get())},
        {"SCT",read(sct_.resource.Get())},{"C24",read(cb_.resource.Get())},{"root",read(root_.resource.Get())},
        {"Selected",read(e.selected_bits.Get())},{"Cover",read(e.cover_bits.Get())},{"BODY",body()},
        {"Inside",read(f.inside_bits.Get())},{"Family",read(e.family_models.Get())},
        {"Code1",matte()},{"RawM",ring_matte()}};
      return o;
    }

    void expect_foreground_delta(const foreground_observation_t &before,const foreground_observation_t &after) {
      ASSERT_EQ(before.mask.size(),after.mask.size());ASSERT_EQ(before.data.size(),68608u);ASSERT_EQ(after.data.size(),before.data.size());
      EXPECT_EQ(before.cbc,after.cbc);EXPECT_EQ(before.raw,after.raw);EXPECT_EQ(before.immutable,after.immutable);
      const auto raw_m=words(before.immutable.at("RawM")),code1=words(before.immutable.at("Code1"));
      ASSERT_EQ(raw_m.size(),before.mask.size());ASSERT_EQ(code1.size(),before.mask.size());
      UINT raw_added=0u,raw_recovered=0u,code1_recovered=0u;
      for(std::size_t i=0u;i<before.mask.size();++i) {
        ASSERT_LE(before.mask[i],1u);ASSERT_LE(after.mask[i],1u);EXPECT_LE(before.mask[i],after.mask[i]);
        if(before.mask[i]==0u&&after.mask[i]==1u) {
          if(raw_m[i]==0u)++raw_added;else ++raw_recovered;
          if(code1[i]!=0u)++code1_recovered;
        }
      }
      ASSERT_GE(before.data[73u],raw_recovered);ASSERT_GE(before.data[76u],code1_recovered);
      EXPECT_EQ(after.data[7u],before.data[7u]+raw_added);EXPECT_EQ(after.data[70u],before.data[70u]+raw_added);
      EXPECT_EQ(after.data[73u],before.data[73u]-raw_recovered);EXPECT_EQ(after.data[76u],before.data[76u]-code1_recovered);
      for(std::size_t i=0u;i<before.data.size();++i)
        if(i!=7u&&i!=70u&&i!=73u&&i!=76u)EXPECT_EQ(after.data[i],before.data[i])<<"F2 word "<<i;
    }

    // Independently authored source AA corner: no per-pixel ownership/model
    // labels are injected. The usual GPU producer must establish every role.
    UINT author_foreground_corner(bool overlap,UINT frame) {
      foreground_edge_enabled(false);
      author(true,frame,3u,false,false,overlap,64u,32u,3u);
      // Independent brighter black-neighbor population inside the OCR core.
      // Uniform nonliteral RGB3 anchors are deliberately UNKNOWN; this small
      // white interior with a gray32 rim makes the ordinary palette learner
      // choose a genuine type1/2 dark cluster. No model records are uploaded.
      rect(44u,12u,48u,24u,{32u,32u,32u,255u});
      rect(46u,14u,47u,22u,{255u,255u,255u,255u});
      rect(50u,12u,55u,18u,{255u,255u,255u,255u});
      constexpr UINT qx=55u,qy=18u;
      // Original F=255,K=3, alpha128, integer encoded-RGB source composition.
      const std::uint8_t code=std::uint8_t((128u*255u+127u*3u+127u)/255u);
      origin_pixels_[qy*W_+qx]={code,code,code,255u};
      upload_pixels_from_origin();return qy*W_+qx;
    }

    // Explicit OWNER-ONLY API fixture. These complete model/membership records
    // are authored, not detector output. The nine Saved production replays own
    // producer-to-FG parity; this seam isolates snapshot, overlap and accounting.
    struct foreground_record_seam_t {
      models::host_sbs_current_foreground_edge_gpu::resources_t owner;
      texture_t selected,cover,family,inside,body,mask,raw_m,code1;
      buffer_t cbc,shape,args;
      ComPtr<ID3D11Buffer> data;
      ComPtr<ID3D11UnorderedAccessView> data_uav,mask_uav;
      std::vector<UINT> original,initial_data;
      UINT q=0u,added=0u,next=0u,owners=0u;
    };

    foreground_record_seam_t foreground_record_seam(bool overlap,UINT frame) {
      author(true,frame,3u,false,false,overlap,64u,32u,3u);
      foreground_record_seam_t s;s.owners=overlap?3u:1u;
      const UINT p=15u*W_+23u,second=15u*W_+29u;
      s.q=16u*W_+24u;s.next=17u*W_+25u;s.added=16u*W_+30u;
      origin_pixels_.assign(std::size_t(W_)*H_,{32u,32u,32u,255u});
      origin_pixels_[p]=origin_pixels_[second]={255u,255u,255u,255u};
      origin_pixels_[s.q]=origin_pixels_[s.added]={129u,129u,129u,255u};
      origin_pixels_[s.next]={65u,65u,65u,255u};upload_pixels_from_origin();
      s.selected=texture(W_,H_,DXGI_FORMAT_R32_UINT);s.cover=texture(W_,H_,DXGI_FORMAT_R32_UINT);
      s.family=texture(17u,8u,DXGI_FORMAT_R32_UINT);s.inside=texture(W_,H_,DXGI_FORMAT_R32_UINT);
      s.body=texture(W_,H_,DXGI_FORMAT_R32_UINT);s.mask=texture(W_,H_,DXGI_FORMAT_R32_UINT);
      s.raw_m=texture(W_,H_,DXGI_FORMAT_R32_UINT);s.code1=texture(W_,H_,DXGI_FORMAT_R32_UINT);
      const std::size_t n=std::size_t(W_)*H_;std::vector<UINT> selected(n),cover(n),zero(n),raw_m(n),code1(n);
      s.original.assign(n,0u);s.original[p]=s.original[second]=1u;
      selected[p]=selected[second]=selected[s.q]=s.owners;
      // q is an original LOW/M0 exact-family colour. If newly emitted M were
      // incorrectly read as a donor, it could supply the next pixel. It must not.
      for(UINT q:{p,second,s.q,s.added,s.next})cover[q]=s.owners;
      raw_m[s.q]=code1[s.q]=1u;
      auto image=[&](texture_t &t,const std::vector<UINT>& v){context_->UpdateSubresource(t.resource.Get(),0u,nullptr,v.data(),W_*4u,0u);};
      image(s.selected,selected);image(s.cover,cover);image(s.inside,zero);image(s.body,zero);
      image(s.mask,s.original);image(s.raw_m,raw_m);image(s.code1,code1);
      std::vector<UINT> meta(512u),families(17u*8u);
      meta[0u]=0x31434e45u;meta[1u]=5u;meta[2u]=meta[4u]=meta[5u]=meta[9u]=1u;
      std::copy(cb_words_.begin(),cb_words_.end(),meta.begin()+16u);
      meta[80u]=sct_words_[3u];meta[81u]=0x31424649u;meta[82u]=2u;meta[83u]=17u;meta[84u]=8u;
      meta[85u]=meta[86u]=s.owners;
      for(UINT unit=0u;unit<(overlap?2u:1u);++unit) {
        const UINT u=128u+32u*unit,b=384u+16u*unit;
        meta[u]=meta[u+1u]=meta[u+29u]=1u;meta[u+26u]=W_;meta[u+27u]=H_;
        // Complete syntactically valid learned type1 K model: two RGB3 votes
        // in three anchors, supported in both sampled columns, unique dark cut.
        const std::array<UINT,16u> k{0x314b4c42u,1u,1u,1u,3u,0x030303u,2u,2u,2u,2u,3u,0x030303u,0x030303u,512u,0u,0u};
        std::copy(k.begin(),k.end(),meta.begin()+b);
        families[17u*unit]=0x0101u;families[17u*unit+1u]=0x646464u;families[17u*unit+2u]=0xffffffu;
      }
      s.cbc=buffer(512u*4u);context_->UpdateSubresource(s.cbc.resource.Get(),0u,nullptr,meta.data(),0u,0u);
      context_->UpdateSubresource(s.family.resource.Get(),0u,nullptr,families.data(),17u*4u,0u);
      s.initial_data.assign(68608u,0u);s.initial_data[0u]=0x37424643u;s.initial_data[1u]=7u;s.initial_data[2u]=1u;
      std::copy(cb_words_.begin(),cb_words_.end(),s.initial_data.begin()+8u);
      s.initial_data[7u]=s.initial_data[70u]=2u;s.initial_data[73u]=s.initial_data[76u]=1u;
      s.initial_data[77u]=0x31444e49u;s.initial_data[78u]=s.initial_data[83u]=1u;
      D3D11_BUFFER_DESC dd{};dd.ByteWidth=68608u*4u;dd.Usage=D3D11_USAGE_DEFAULT;
      dd.BindFlags=D3D11_BIND_UNORDERED_ACCESS;dd.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
      require(SUCCEEDED(device_->CreateBuffer(&dd,nullptr,&s.data)));
      context_->UpdateSubresource(s.data.Get(),0u,nullptr,s.initial_data.data(),0u,0u);
      D3D11_UNORDERED_ACCESS_VIEW_DESC ud{};ud.Format=DXGI_FORMAT_R32_TYPELESS;ud.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;
      ud.Buffer.NumElements=68608u;ud.Buffer.Flags=D3D11_BUFFER_UAV_FLAG_RAW;
      require(SUCCEEDED(device_->CreateUnorderedAccessView(s.data.Get(),&ud,&s.data_uav)));
      s.mask_uav=uav(s.mask.srv.Get());s.shape=buffer(16u,true);s.args=buffer(12u,false,true);
      const std::array<UINT,4u> shape{W_,H_,W_*H_,0u};const std::array<UINT,3u> args{(W_+15u)/16u,(H_+15u)/16u,1u};
      context_->UpdateSubresource(s.shape.resource.Get(),0u,nullptr,shape.data(),0u,0u);
      context_->UpdateSubresource(s.args.resource.Get(),0u,nullptr,args.data(),0u,0u);
      require(s.owner.ensure(device_.Get(),W_,H_,s.mask.resource.Get(),s.mask_uav.Get(),error_));return s;
    }

    void run_foreground_record_seam(foreground_record_seam_t &s,bool enabled=true,bool work=true) {
      const std::array<UINT,3u> args{(W_+15u)/16u,(H_+15u)/16u,work?1u:0u};
      context_->UpdateSubresource(s.args.resource.Get(),0u,nullptr,args.data(),0u,0u);
      models::host_sbs_current_foreground_edge_gpu::inputs_t in;
      in.rgb=rgb_.srv.Get();in.ocr=ocr_.srv.Get();in.selected=s.selected.srv.Get();in.cover=s.cover.srv.Get();
      in.cbc=s.cbc.srv.Get();in.models=s.family.srv.Get();in.sct=sct_.srv.Get();
      in.inside=s.inside.srv.Get();in.body=s.body.srv.Get();in.raw_m=s.raw_m.srv.Get();in.code1=s.code1.srv.Get();
      in.original_final=s.mask.resource.Get();in.final_m=s.mask_uav.Get();in.data=s.data_uav.Get();
      const models::host_sbs_current_foreground_edge_gpu::programs_t p{enabled?foreground_edge_shaders_[0u].Get():nullptr};
      s.owner.finish(context_.Get(),p,cb_.resource.Get(),s.shape.resource.Get(),in,s.args.resource.Get());
    }

    std::map<std::string,std::vector<char>> foreground_record_immutable(const foreground_record_seam_t &s) {
      return {{"RGB",read(rgb_.resource.Get())},{"OCR",read(ocr_.resource.Get())},{"SCT",read(sct_.resource.Get())},
        {"C24",read(cb_.resource.Get())},{"CBC",read(s.cbc.resource.Get())},{"Family",read(s.family.resource.Get())},
        {"Selected",read(s.selected.resource.Get())},{"Cover",read(s.cover.resource.Get())},
        {"Inside",read(s.inside.resource.Get())},{"BODY",read(s.body.resource.Get())},
        {"Raw",read(s.raw_m.resource.Get())},{"Code1",read(s.code1.resource.Get())}};
    }

    std::array<ComPtr<ID3D11ComputeShader>,1u> foreground_edge_shaders_;
    std::string foreground_edge_closure_;
