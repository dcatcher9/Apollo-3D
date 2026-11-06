// Independent route1 source diagnostic. It never modifies the original strict
// route0 diagnostic or grants Raw/black/model/render authority to TIR bits.
if (trusted_diagnostics) {
  ASSERT_TRUE(current) << "Current production selector receipt is required before TIR interpretation";
  ASSERT_EQ(o[4u],1u) << "Trusted-source diagnostic owns exactly current unit0";
  ASSERT_TRUE(bundle.enclosure.trusted_inner_bits);
  ASSERT_TRUE(bundle.enclosure.trusted_publication_arguments);
  arrays.emplace("trusted_analysis_RGBA16F.bin",read(bundle.source.analysis_band.Get()));
  const auto reference=read(bundle.source.reference_rgb8.Get());
  const auto band_first=std::size_t(shape[0])*shape[2]*4u;
  ASSERT_EQ(reference.size(),std::size_t(shape[0])*shape[1]*4u);
  arrays.emplace("trusted_reference_band_RGBA8.bin",std::vector<char>(reference.begin()+band_first,reference.begin()+band_first+pixels*4u));
  const auto diagnostic_source=fs::path(SUNSHINE_SOURCE_DIR)/"tests/unit/current_outline_trusted_inner_saved.inc";
  const auto diagnostic_source_bytes=file_bytes(diagnostic_source);
  baseline::resources_t trusted_owner;
  ASSERT_TRUE(trusted_owner.ensure(device_.Get(),shape[0],shape[3],error_)) << error_;
  ASSERT_TRUE(trusted_owner.dispatch(context_.Get(),effective_composition.colour,effective_composition.components,
    bundle.current.actual_contour_constants96.Get(),bundle.source.analysis_band.Get(),
    bundle.current.probability.Get(),bundle.current.support.Get(),bundle.current.ocr_record.Get(),
    bundle.current.actual_final256.Get(),240u,error_)) << error_;
  const auto b=trusted_owner.mutable_views();
  arrays.emplace("trusted_diagnostic_original_BODY.u32",read(b.body.Get()));
  arrays.emplace("trusted_diagnostic_numeric_proof.u32",read(b.validation_receipt.Get()));
  const bool body_parity=arrays.at("trusted_diagnostic_original_BODY.u32")==arrays.at("BODY.u32");
  const bool proof_parity=arrays.at("trusted_diagnostic_numeric_proof.u32")==arrays.at("numeric_proof.u32");
  EXPECT_TRUE(body_parity);EXPECT_TRUE(proof_parity);
  trusted_report={{"role","trusted-inner-source-diagnostic"},{"unit",0u},{"route",1u},
    {"source","Exact immutable production Source5 RGB/P/S/OCR/C96/root; independent scratch owner"},
    {"fixture_source",{{"path",diagnostic_source.string()},{"sha256",hash_bytes(diagnostic_source_bytes)}}},
    {"captured_selected_matte_byteexact",exact},{"original_BODY_byteexact",body_parity},
    {"numeric_proof_byteexact",proof_parity},{"interpretation_authorized",false},
    {"history_restored",false},{"rendering_authority",false},{"modes",json::array()}};
  if(body_parity&&proof_parity) {
    auto hook=buffer(16u,true),graph_shape=buffer(16u,true);
    const std::array<UINT,4u> dimensions{shape[0],shape[3],UINT(pixels),0u},unit0{0u,0u,1u,0u};
    context_->UpdateSubresource(hook.resource.Get(),0u,nullptr,unit0.data(),0u,0u);
    context_->UpdateSubresource(graph_shape.resource.Get(),0u,nullptr,dimensions.data(),0u,0u);
    const auto unbind=[&]() {
      std::array<ID3D11ShaderResourceView *,16u> in{};
      std::array<ID3D11UnorderedAccessView *,8u> out{};
      context_->CSSetShaderResources(0u,UINT(in.size()),in.data());
      context_->CSSetUnorderedAccessViews(0u,UINT(out.size()),out.data(),nullptr);
    };
    auto palette_uav=uav(b.palette.Get());
    D3D11_UNORDERED_ACCESS_VIEW_DESC args_desc{};
    args_desc.Format=DXGI_FORMAT_R32_TYPELESS;args_desc.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;
    args_desc.Buffer.NumElements=24u;args_desc.Buffer.Flags=D3D11_BUFFER_UAV_FLAG_RAW;
    ComPtr<ID3D11UnorderedAccessView> args_uav;
    ASSERT_TRUE(SUCCEEDED(device_->CreateUnorderedAccessView(b.current_mode_args.Get(),&args_desc,&args_uav)));
    std::array<ID3D11Buffer *,3u> colour_constants{bundle.current.actual_contour_constants96.Get(),hook.resource.Get(),nullptr};
    std::array<ID3D11ShaderResourceView *,9u> colour_inputs{bundle.source.analysis_band.Get(),bundle.current.probability.Get(),
      bundle.current.support.Get(),bundle.current.ocr_record.Get(),nullptr,nullptr,nullptr,nullptr,b.validation_receipt.Get()};
    std::array<ID3D11UnorderedAccessView *,8u> outputs{};
    unbind();constexpr std::array<UINT,4u> zero{};
    context_->ClearUnorderedAccessViewUint(args_uav.Get(),zero.data());
    outputs[0u]=palette_uav.Get();outputs[7u]=args_uav.Get();
    context_->CSSetConstantBuffers(0u,UINT(colour_constants.size()),colour_constants.data());
    context_->CSSetShaderResources(0u,UINT(colour_inputs.size()),colour_inputs.data());
    context_->CSSetUnorderedAccessViews(0u,UINT(outputs.size()),outputs.data(),nullptr);
    context_->CSSetShader(effective_composition.colour[0u],nullptr,0u);
    context_->DispatchIndirect(bundle.current.actual_final256.Get(),240u);unbind();
    arrays.emplace("trusted_unit0_Palette128.u32",read(b.palette.Get()));
    arrays.emplace("trusted_unit0_ModeArguments.u32",read(b.current_mode_args.Get()));
    const auto palette=words(arrays.at("trusted_unit0_Palette128.u32"));
    const auto mode_arguments=words(arrays.at("trusted_unit0_ModeArguments.u32"));
    ASSERT_EQ(palette.size(),128u);ASSERT_EQ(mode_arguments.size(),24u);
    ASSERT_EQ(palette[0u],0x31524954u);ASSERT_EQ(palette[1u],2u);ASSERT_EQ(palette[2u],1u);
    ASSERT_EQ(palette[3u],0u);ASSERT_LE(palette[4u],1u);ASSERT_EQ(palette[7u],1u);
    ASSERT_EQ(palette[22u],2u);ASSERT_EQ(palette[23u],1u);
    ASSERT_LE(palette[48u],256u);ASSERT_LE(palette[49u],palette[48u]);
    ASSERT_EQ(palette[50u],palette[4u]);ASSERT_LE(palette[51u],2u);
    for(UINT word=53u;word<64u;++word)ASSERT_EQ(palette[word],0u);
    if(palette[4u]==1u) {
      ASSERT_EQ(palette[51u],1u);ASSERT_EQ(palette[48u]-palette[49u],1u);
      ASSERT_EQ(palette[52u],palette[68u]);ASSERT_GE(palette[52u],2u);ASSERT_LE(palette[52u],512u);
    } else {
      ASSERT_NE(palette[51u],1u);
      if(palette[51u]==0u) {ASSERT_EQ(palette[48u],palette[49u]);ASSERT_EQ(palette[52u],0u);}
      else {ASSERT_EQ(palette[48u]-palette[49u],2u);ASSERT_GE(palette[52u],2u);ASSERT_LE(palette[52u],512u);}
    }
    for(UINT word=64u+8u*palette[4u];word<128u;++word)ASSERT_EQ(palette[word],0u);
    for(UINT word=3u*palette[4u];word<24u;++word)ASSERT_EQ(mode_arguments[word],0u);
    colour_inputs[4u]=b.palette.Get();
    std::array<ComPtr<ID3D11UnorderedAccessView>,8u> scratch{uav(b.eligible.Get()),uav(b.seeds.Get()),
      uav(b.domain_tag.Get()),uav(b.cover_clip.Get()),uav(b.parent.Get()),uav(b.root_flags.Get()),
      uav(b.mode_body.Get()),uav(b.mode_final.Get())};
    const std::array<ID3D11ShaderResourceView *,4u> graph_inputs{b.eligible.Get(),b.seeds.Get(),b.domain_tag.Get(),b.cover_clip.Get()};
    const std::array<ID3D11Buffer *,3u> graph_constants{graph_shape.resource.Get(),nullptr,nullptr};
    std::vector<UINT> body_union(pixels,0u);
    for(UINT mode=0u;mode<palette[4u];++mode) {
      const auto prefix="trusted_unit0_mode"+std::to_string(mode)+"_";
      const std::array<UINT,3u> dispatch{mode_arguments[mode*3u],mode_arguments[mode*3u+1u],mode_arguments[mode*3u+2u]};
      if(dispatch[2u]==0u) {
        ASSERT_EQ(dispatch,(std::array<UINT,3u>{0u,0u,0u}));
        trusted_report["modes"].push_back({{"mode",mode},{"dispatched",false},{"dispatch",dispatch}});continue;
      }
      ASSERT_EQ(dispatch,(std::array<UINT,3u>{(shape[0]+15u)/16u,(shape[3]+15u)/16u,1u}));
      const std::array<UINT,4u> request{0u,mode,1u,0u};
      context_->UpdateSubresource(hook.resource.Get(),0u,nullptr,request.data(),0u,0u);
      outputs.fill(nullptr);for(UINT lane=0u;lane<4u;++lane)outputs[1u+lane]=scratch[lane].Get();
      context_->CSSetConstantBuffers(0u,UINT(colour_constants.size()),colour_constants.data());
      context_->CSSetShaderResources(0u,UINT(colour_inputs.size()),colour_inputs.data());
      context_->CSSetUnorderedAccessViews(0u,UINT(outputs.size()),outputs.data(),nullptr);
      context_->CSSetShader(effective_composition.colour[1u],nullptr,0u);
      context_->DispatchIndirect(b.current_mode_args.Get(),mode*12u);unbind();
      arrays.emplace(prefix+"Eligible.u32",read(b.eligible.Get()));arrays.emplace(prefix+"Seeds.u32",read(b.seeds.Get()));
      arrays.emplace(prefix+"Domain.u32",read(b.domain_tag.Get()));arrays.emplace(prefix+"Cover.u32",read(b.cover_clip.Get()));
      outputs.fill(nullptr);for(UINT lane=0u;lane<4u;++lane)outputs[lane]=scratch[4u+lane].Get();
      context_->CSSetConstantBuffers(0u,UINT(graph_constants.size()),graph_constants.data());
      context_->CSSetShaderResources(0u,UINT(graph_inputs.size()),graph_inputs.data());
      context_->CSSetUnorderedAccessViews(0u,UINT(outputs.size()),outputs.data(),nullptr);
      // Trusted production uses exactly the four component passes, without BOX3.
      for(UINT entrypoint:{0u,1u,2u,3u}) {
        context_->CSSetShader(effective_composition.components[entrypoint],nullptr,0u);
        context_->DispatchIndirect(b.current_mode_args.Get(),mode*12u);
      }
      unbind();
      arrays.emplace(prefix+"Parent.u32",read(b.parent.Get()));arrays.emplace(prefix+"RootFlags.u32",read(b.root_flags.Get()));
      arrays.emplace(prefix+"BODY.u32",read(b.mode_body.Get()));arrays.emplace(prefix+"Final.u32",read(b.mode_final.Get()));
      const auto body=words(arrays.at(prefix+"BODY.u32"));ASSERT_EQ(body.size(),pixels);
      std::size_t body_count=0u;
      for(std::size_t pixel=0u;pixel<pixels;++pixel) {ASSERT_LE(body[pixel],1u);body_union[pixel]|=body[pixel];body_count+=body[pixel];}
      trusted_report["modes"].push_back({{"mode",mode},{"BODY_pixels",body_count},
        {"dispatched",true},{"dispatch",dispatch},{"argument_offset",mode*12u}});
    }
    arrays.emplace("trusted_unit0_BODY_union.u32",raw(body_union));
    const bool union_parity=arrays.at("trusted_unit0_BODY_union.u32")==arrays.at("trusted_inner_bits.u32");
    EXPECT_TRUE(union_parity) << "Trusted per-mode BODY union differs from actual production TIR bits";
    trusted_report["BODY_union_byteexact_production_TIR"]=union_parity;
    trusted_report["interpretation_authorized"]=union_parity;
  }
  const auto reference_after=read(bundle.source.reference_rgb8.Get());ASSERT_EQ(reference_after.size(),reference.size());
  const std::map<std::string,std::vector<char>> production_after{
    {"source_proof.u32",read(bundle.source.source_proof.Get())},{"source_envelope.u32",read(bundle.source.uip_envelope.Get())},
    {"CBC.u32",read(bundle.enclosure.receipt.Get())},{"Raw.u32",read(bundle.ring.receipt.Get())},
    {"F2.u32",read(bundle.outline.calibration_data.Get())},{"BODY.u32",read(bundle.enclosure.body.Get())},
    {"CBC_M.u32",read(bundle.enclosure.matte.Get())},{"Raw_M.u32",read(bundle.ring.matte.Get())},
    {"F2_M.u32",read(bundle.outline.matte.Get())},{"selected_M.u32",read(selected_views.matte.Get())},
    {"selected_receipt.u32",read(selected_views.receipt.Get())},{"numeric_proof.u32",read(bundle.numeric_proof.Get())},
    {"trusted_inner_bits.u32",read(bundle.enclosure.trusted_inner_bits.Get())},
    {"trusted_publication_arguments.u32",read(bundle.enclosure.trusted_publication_arguments.Get())},
    {"trusted_analysis_RGBA16F.bin",read(bundle.source.analysis_band.Get())},
    {"trusted_reference_band_RGBA8.bin",std::vector<char>(reference_after.begin()+band_first,reference_after.begin()+band_first+pixels*4u)}};
  bool unchanged=reference_after==reference;
  EXPECT_TRUE(unchanged) << "Trusted diagnostic modified the full Source5 reference resource";
  trusted_report["full_reference_resource_byteexact"]=unchanged;
  for(const auto &[name,bytes]:production_after) {
    const bool same=bytes==arrays.at(name);EXPECT_TRUE(same) << "Trusted diagnostic modified " << name;
    trusted_report["production_after_byteexact"][name]=same;unchanged=unchanged&&same;
  }
  trusted_report["interpretation_authorized"]=trusted_report.at("interpretation_authorized").get<bool>()&&unchanged;
  ASSERT_EQ(file_bytes(diagnostic_source),diagnostic_source_bytes);
}
