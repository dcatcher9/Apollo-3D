// Enabled when the explicit canonical FG shader root/pin is supplied.
TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, ForegroundEdgeOwnedCornerAndOverlapUseUniqueCurrentAccounting) {
  if(foreground_edge_closure_.empty())GTEST_SKIP()<<"Explicit foreground-edge root/pin required";
  for(bool overlap:{false,true}) {
    auto s=foreground_record_seam(overlap,11000u+UINT(overlap));
    const auto immutable=foreground_record_immutable(s);
    ASSERT_EQ(words(read(s.mask.resource.Get())),s.original);
    run_foreground_record_seam(s);
    const auto mask=words(read(s.mask.resource.Get())),data=words(read(s.data.Get()));
    auto expected=s.original;expected[s.q]=expected[s.added]=1u;
    EXPECT_EQ(mask,expected);EXPECT_EQ(mask[s.next],0u);
    EXPECT_EQ(words(read(s.owner.original_mask())),s.original);
    EXPECT_EQ(foreground_record_immutable(s),immutable);
    auto expected_data=s.initial_data;expected_data[7u]=expected_data[70u]=3u;
    expected_data[73u]=expected_data[76u]=0u;EXPECT_EQ(data,expected_data);
    RecordProperty(overlap?"FG_overlap_scope":"FG_corner_scope",
      "OWNER-ONLY authored complete current records: no detector/producer readiness claim. Actual FG shader; immutable original M, no recursive LOW/M1 donor, overlap counted once, Raw/Code1 recovery and new-support accounting.");
  }
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, ForegroundEdgeSnapshotContainsActualPostHoleMaskAndNeverFeedback) {
  if(foreground_edge_closure_.empty()||hole_closure_.empty()||coverage_closure_.empty())
    GTEST_SKIP()<<"Explicit FG plus current C7/G1 programs required";
  foreground_edge_enabled(false);
  ASSERT_NO_FATAL_FAILURE(expect_authored_low_contrast_aa(false,64u,5u,11002u,"fg_posthole_snapshot_"));
  const auto before=foreground_observation();const auto proof=words(read(helper_.mask_hole_proof()));
  const UINT q=63u*W_+75u;ASSERT_EQ(before.mask[q],1u);ASSERT_EQ(proof[q]&0x10000u,0u);
  foreground_edge_enabled(true);ASSERT_TRUE(run())<<error_;
  const auto snapshot=words(read(helper_.foreground_edge_original_mask()));
  EXPECT_EQ(snapshot,before.mask);EXPECT_EQ(snapshot[q],1u);
  EXPECT_EQ(words(read(helper_.mask_hole_proof())),proof);
  const auto after=foreground_observation();expect_foreground_delta(before,after);
  // Every newly emitted pixel remains zero in the complete immutable donor image.
  for(std::size_t i=0u;i<snapshot.size();++i)if(before.mask[i]==0u&&after.mask[i]!=0u)EXPECT_EQ(snapshot[i],0u);
  RecordProperty("FG_snapshot_scope","Actual post-C7/G1 M differs from pre-hole proof bit16; full donor image equals pre-FG M");
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, ForegroundEdgeMissingProgramRootZeroAndStaleContextPreserveFallback) {
  if(foreground_edge_closure_.empty())GTEST_SKIP()<<"Explicit foreground-edge root/pin required";
  auto s=foreground_record_seam(false,11003u);
  const auto immutable=foreground_record_immutable(s);
  run_foreground_record_seam(s,false);
  EXPECT_EQ(words(read(s.mask.resource.Get())),s.original);EXPECT_EQ(words(read(s.data.Get())),s.initial_data);
  run_foreground_record_seam(s);
  const auto active=words(read(s.mask.resource.Get())),data=words(read(s.data.Get()));
  ASSERT_EQ(active[s.q],1u);ASSERT_EQ(active[s.added],1u);
  run_foreground_record_seam(s,true,false); // Actual zero-depth indirect dispatch.
  EXPECT_EQ(words(read(s.mask.resource.Get())),active);EXPECT_EQ(words(read(s.data.Get())),data);
  EXPECT_EQ(foreground_record_immutable(s),immutable);
  // Reset only the controlled seam operands, then give the shader a stale SCT.
  context_->UpdateSubresource(s.mask.resource.Get(),0u,nullptr,s.original.data(),W_*4u,0u);
  context_->UpdateSubresource(s.data.Get(),0u,nullptr,s.initial_data.data(),0u,0u);
  sct_words_[4u]^=1u;upload();const auto stale_inputs=foreground_record_immutable(s);
  run_foreground_record_seam(s);
  EXPECT_EQ(words(read(s.mask.resource.Get())),s.original);EXPECT_EQ(words(read(s.data.Get())),s.initial_data);
  EXPECT_EQ(foreground_record_immutable(s),stale_inputs);
  RecordProperty("FG_fallback_scope","OWNER-ONLY authored current records; missing optional program, real Root0 indirect hold, stale-context no admission. Full producer fallback is covered separately by nine Saved replays and the foreign-device regression.");
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, ForegroundEdgeRestoresAllFourConstantBufferSlots) {
  if(foreground_edge_closure_.empty())GTEST_SKIP()<<"Explicit foreground-edge root/pin required";
  author_foreground_corner(false,11004u);foreground_edge_enabled(true);
  std::array<ID3D11Buffer*,4u> expected{cb_.resource.Get(),cb_.resource.Get(),cb_.resource.Get(),cb_.resource.Get()};
  std::array<ID3D11ShaderResourceView*,16u> sources{};sources.fill(rgb_.srv.Get());
  auto dummy=texture(W_,H_,DXGI_FORMAT_R32_UINT);auto output=uav(dummy.srv.Get());
  ID3D11UnorderedAccessView *expected_uav=output.Get();
  context_->CSSetShaderResources(0u,16u,sources.data());context_->CSSetUnorderedAccessViews(7u,1u,&expected_uav,nullptr);
  context_->CSSetShader(colour_[0u].Get(),nullptr,0u);
  context_->CSSetConstantBuffers(0u,4u,expected.data());ASSERT_TRUE(run(true,false))<<error_;
  std::array<ID3D11Buffer*,4u> actual{};context_->CSGetConstantBuffers(0u,4u,actual.data());
  for(UINT i=0u;i<actual.size();++i){EXPECT_EQ(actual[i],expected[i]);if(actual[i])actual[i]->Release();}
  std::array<ID3D11ShaderResourceView*,16u> restored{};context_->CSGetShaderResources(0u,16u,restored.data());
  for(UINT i=0u;i<restored.size();++i){EXPECT_EQ(restored[i],sources[i]);if(restored[i])restored[i]->Release();}
  ID3D11UnorderedAccessView *restored_uav=nullptr;context_->CSGetUnorderedAccessViews(7u,1u,&restored_uav);
  EXPECT_EQ(restored_uav,expected_uav);if(restored_uav)restored_uav->Release();
  ID3D11ComputeShader *shader=nullptr;UINT n=0u;context_->CSGetShader(&shader,nullptr,&n);
  EXPECT_EQ(shader,colour_[0u].Get());if(shader)shader->Release();
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, ForegroundEdgeForeignOptionalShaderCannotReplaceCurrentMask) {
  if(foreground_edge_closure_.empty())GTEST_SKIP()<<"Explicit foreground-edge root/pin required";
  author_foreground_corner(false,11005u);ASSERT_TRUE(run())<<error_;const auto before=foreground_observation();
  ComPtr<ID3D11Device> foreign;ComPtr<ID3D11DeviceContext> foreign_context;
  const D3D_FEATURE_LEVEL level=D3D_FEATURE_LEVEL_11_0;D3D_FEATURE_LEVEL actual{};
  ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_WARP,nullptr,0u,&level,1u,D3D11_SDK_VERSION,&foreign,&actual,&foreign_context)));
  const auto &spec=models::current_outline_runtime::foreground_edge_specs;
  const auto source=cache::snapshot_sources(std::getenv("SUNSHINE_TEST_FOREGROUND_EDGE_SHADER_ROOT"),spec);
  ASSERT_EQ(cache::source_closure_sha256(source),foreground_edge_closure_);
  const auto code=cache::get(source,spec[0u]);ASSERT_TRUE(code&&!code->empty());
  ComPtr<ID3D11ComputeShader> wrong;ASSERT_TRUE(SUCCEEDED(foreign->CreateComputeShader(code->data(),code->size(),nullptr,&wrong)));
  program_.foreground_edge_shaders[0u]=wrong.Get();ASSERT_TRUE(run())<<error_;
  const auto after=foreground_observation();EXPECT_EQ(after.mask,before.mask);EXPECT_EQ(after.data,before.data);
  EXPECT_EQ(after.immutable,before.immutable);EXPECT_EQ(helper_.foreground_edge_original_mask(),nullptr);
}
