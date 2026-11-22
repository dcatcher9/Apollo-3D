// PRIVATE opt-in two-axis geometry experiment. This include is not a live path.
namespace {
  struct private_two_axis_files_t {
    using json = nlohmann::json;
    static void require(bool value, const std::string &message) {
      if (!value) throw std::runtime_error(message);
    }
    static std::vector<std::byte> read(const std::filesystem::path &path) {
      std::ifstream stream(path, std::ios::binary | std::ios::ate);
      require(bool(stream) && stream.tellg() >= 0, "Cannot read private two-axis operand: " + path.string());
      std::vector<std::byte> bytes(static_cast<std::size_t>(stream.tellg()));stream.seekg(0);
      stream.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
      require(bool(stream), "Incomplete private two-axis operand: " + path.string());return bytes;
    }
    static std::string sha(const std::vector<std::byte> &bytes) {
      const auto hash = crypto::hash(std::string_view(reinterpret_cast<const char *>(bytes.data()), bytes.size()));
      constexpr char digits[] = "0123456789abcdef";std::string value;
      for (auto byte : hash) {value.push_back(digits[byte >> 4u]);value.push_back(digits[byte & 15u]);}
      return value;
    }
    static json binding(const std::filesystem::path &path) {
      const auto bytes=read(path);return {{"path",path.string()},{"byte_count",bytes.size()},{"sha256",sha(bytes)}};
    }
    static json document(const std::vector<std::byte> &bytes) {
      return json::parse(std::string_view(reinterpret_cast<const char *>(bytes.data()),bytes.size()));
    }
    template<class T> static std::vector<T> load(const json &spec, std::size_t count) {
      const auto bytes=read(spec.at("path").get<std::string>());
      require(bytes.size()==count*sizeof(T) && spec.at("byte_count")==bytes.size() && spec.at("sha256")==sha(bytes),
              "Private two-axis bound operand changed");
      std::vector<T> out(count);std::memcpy(out.data(),bytes.data(),bytes.size());return out;
    }
    static bool configure(live_v2_warp_fixture_t &fixture, std::string &error) {
      const auto *root=std::getenv("SUNSHINE_TEST_PRIVATE_TWO_AXIS_ROOT");
      const auto *pin=std::getenv("SUNSHINE_TEST_PRIVATE_TWO_AXIS_PIN");
      if (!root || !*root || !pin || !*pin) {error="Explicit staged private two-axis root and closure required";return false;}
      return fixture.configure_private_two_axis(root,pin,error);
    }
  };
}

TEST(HostSbsV2LiveWarpGpuTest, PrivateTwoAxisPinDistanceGeometry) {
  if (!std::getenv("SUNSHINE_TEST_PRIVATE_TWO_AXIS_ROOT")) GTEST_SKIP()<<"No staged private two-axis program selected";
  constexpr UINT W=128u,H=32u;std::string error;live_v2_warp_fixture_t fixture;
  ASSERT_TRUE(fixture.initialize(error,warp_shader_e::selected_outline,false,true))<<error;
  ASSERT_TRUE(private_two_axis_files_t::configure(fixture,error))<<error;
  const auto state=make_live_state(true,true);const float plane=3.0f/W;
  for (const std::string kind : {"empty","single","row","edges","mixed-base"}) {
    SCOPED_TRACE(kind);auto ui=make_subtitle_inputs(W,H,state,plane);
    std::vector<std::pair<UINT,UINT>> seeds;
    if(kind=="single" || kind=="mixed-base") seeds={{W/2u,H/2u}};
    if(kind=="row")for(UINT x=37u;x<91u;++x)seeds.emplace_back(x,H/2u);
    if(kind=="edges")seeds={{0u,0u},{W-1u,0u},{0u,H-1u},{W-1u,H-1u}};
    std::vector<rgba16_bits_t> source(W*H,{0x3400u,0x3400u,0x3400u,0x3c00u});
    for(const auto &[x,y]:seeds) {ui.matte[y*W+x]=1u;ui.snapshot[y*W+x]={0x3c00u,0x3c00u,0x3c00u,0x3c00u};source[y*W+x]=ui.snapshot[y*W+x];}
    std::vector<float> base(W*H,-2.0f/W);
    if(kind=="mixed-base")for(UINT y=0u;y<H;++y)for(UINT x=0u;x<W;++x)
      base[y*W+x]=(0.01f*(int(x)-64)+0.8f*std::clamp(int(y)-16,-3,3))/W;
    auto operands=make_current_outline_inputs(ui);operands.private_two_axis_ui_pin=true;
    shared_ui_pin_diagnostics_t diagnostic;operands.shared_ui_pin_diagnostic_output=&diagnostic;
    std::vector<std::byte> pixels;
    ASSERT_TRUE(fixture.render(DXGI_FORMAT_R16G16B16A16_FLOAT,8u,W,H,source.data(),base,base,state,{},pixels,error,W,H,{},&ui,nullptr,true,&operands))<<error;
    ASSERT_EQ(diagnostic.shape,(std::array<UINT,4u>{W,H,0u,H}));
    ASSERT_EQ(diagnostic.private_original_base.size(),base.size());ASSERT_EQ(diagnostic.private_final_distance.size(),base.size());
    ASSERT_TRUE(diagnostic.selected_receipt_authorized);
    double base_y=0.0,final_y=0.0;
    for(UINT y=0u;y<H;++y)for(UINT x=0u;x<W;++x) {
      const auto at=std::size_t(y)*W+x;double bound=1.0;
      for(const auto &[sx,sy]:seeds)bound=std::min(bound,0.5*(std::max(std::abs(int(x)-int(sx))-1,0)+std::max(std::abs(int(y)-int(sy))-1,0))/W);
      // An independent closed-form distance check, not a CPU image renderer.
      EXPECT_NEAR(diagnostic.private_final_distance[at],bound,1e-5/W);
      const double h=diagnostic.private_original_base[at],g=diagnostic.field[at];
      EXPECT_NEAR(g,std::clamp(h,double(plane)-bound,double(plane)+bound),2e-5/W);
      EXPECT_GE(g,std::min(h,double(plane))-2e-5/W);EXPECT_LE(g,std::max(h,double(plane))+2e-5/W);
      if(x)EXPECT_LE(std::abs(g-diagnostic.field[at-1u])*W,0.50002);
      if(y) {base_y=std::max(base_y,std::abs(h-diagnostic.private_original_base[at-W])*W);final_y=std::max(final_y,std::abs(g-diagnostic.field[at-W])*W);}
    }
    EXPECT_LE(final_y,std::max(base_y,0.5)+2e-5)<<"Two-axis cone preserves the larger Base/cone vertical slope; it does not promise 0.5 for a steeper Base";
    if(kind=="empty") {
      EXPECT_EQ(diagnostic.field,diagnostic.private_original_base);
      auto disabled=ui;disabled.constants.enabled=0u;std::vector<std::byte> ordinary;
      ASSERT_TRUE(fixture.render(DXGI_FORMAT_R16G16B16A16_FLOAT,8u,W,H,source.data(),base,base,state,{},ordinary,error,W,H,{},&disabled))<<error;
      EXPECT_EQ(pixels,ordinary);
    }
  }
  EXPECT_TRUE(fixture.private_two_axis_source_unchanged());
}

TEST(HostSbsV2LiveWarpGpuTest, PrivateTwoAxisPinColourAndCurrentAuthority) {
  if (!std::getenv("SUNSHINE_TEST_PRIVATE_TWO_AXIS_ROOT")) GTEST_SKIP()<<"No staged private two-axis program selected";
  constexpr UINT W=128u,H=32u;std::string error;live_v2_warp_fixture_t fixture;
  ASSERT_TRUE(fixture.initialize(error,warp_shader_e::selected_outline,false,true))<<error;
  ASSERT_TRUE(private_two_axis_files_t::configure(fixture,error))<<error;
  const auto state=make_live_state(true,true);const std::vector<float> base(W*H,-1.25f/W);
  for(UINT mode:{0u,1u,2u})for(bool roi:{false,true})for(float scale_y:{1.0f,0.8125f}) {
    SCOPED_TRACE(std::to_string(mode)+"/"+std::to_string(roi)+"/"+std::to_string(scale_y));
    const UINT aw=roi?64u:W,ah=roi?16u:H,left=roi?32u:0u,top=roi?8u:0u;
    auto ui=make_subtitle_inputs(aw,ah,state,2.5f/aw,W,H);ui.constants.color_mode=mode;ui.constants.source_left=left;ui.constants.source_top=top;
    host_sbs_v2_geometry_t geometry;geometry.content_scale_y=scale_y;
    if(roi) {geometry.video_roi_active=1.0f;geometry.video_roi_left=geometry.video_roi_top=0.25f;geometry.video_roi_right=geometry.video_roi_bottom=0.75f;geometry.tensor_content_right=W;geometry.tensor_content_bottom=H;}
    std::vector<rgba16_bits_t> source(W*H,{0x3400u,0x3400u,0x3400u,0x3c00u});
    const std::uint16_t raw=mode==2u?0x3c00u:0x3800u,display=mode==0u?0x3800u:0x39e2u;
    for(UINT y=ah/2u-3u;y<ah/2u+4u;++y)for(UINT x=aw/2u-5u;x<aw/2u+6u;++x) {
      ui.matte[y*aw+x]=1u;ui.snapshot[y*aw+x]={display,display,display,0x3c00u};source[(y+top)*W+x+left]={raw,raw,raw,0x3c00u};
    }
    auto selected=make_current_outline_inputs(ui);selected.private_two_axis_ui_pin=true;
    shared_ui_pin_diagnostics_t diagnostic;selected.shared_ui_pin_diagnostic_output=&diagnostic;
    std::vector<std::byte> actual,reference,ordinary;
    ASSERT_TRUE(fixture.render(DXGI_FORMAT_R16G16B16A16_FLOAT,8u,W,H,source.data(),base,base,state,{},actual,error,W,H,geometry,&ui,nullptr,true,&selected))<<error;
    auto disabled=ui;disabled.constants.enabled=0u;auto flat=geometry;flat.video_roi_active=0.0f;const std::vector<float> constant(W*H,2.5f/W);
    ASSERT_TRUE(fixture.render(DXGI_FORMAT_R16G16B16A16_FLOAT,8u,W,H,source.data(),constant,constant,state,{},reference,error,W,H,flat,&disabled))<<error;
    const auto coverage=unpack_rgba32f(diagnostic.ui_coverage_rgba32f);std::size_t covered=0u;
    for(std::size_t at=0u;at<coverage.size();++at)if(coverage[at].g==1.0f) {++covered;EXPECT_EQ(std::memcmp(actual.data()+at*8u,reference.data()+at*8u,8u),0);}
    EXPECT_GT(covered,0u);
    ASSERT_TRUE(fixture.render(DXGI_FORMAT_R16G16B16A16_FLOAT,8u,W,H,source.data(),base,base,state,{},ordinary,error,W,H,geometry,&disabled))<<error;
    for(const std::string bad_kind:{"empty","stale-sct","revoked-rgb","stale-receipt","invalid-plane"}) {
      SCOPED_TRACE(bad_kind);auto bad=ui;auto operands=selected;operands.shared_ui_pin_diagnostic_output=nullptr;
      if(bad_kind=="empty")std::fill(operands.matte.begin(),operands.matte.end(),0u);
      if(bad_kind=="stale-sct")bad.identity[20u]^=1u;
      if(bad_kind=="revoked-rgb")for(auto &pixel:bad.snapshot)pixel={0u,0x3c00u,0x3c00u,0x3c00u};
      if(bad_kind=="stale-receipt")operands.receipt[48u]^=1u;
      if(bad_kind=="invalid-plane")bad.condition[5u]=0x7fc00000u;
      std::vector<std::byte> rejected,expected=ordinary;
      ASSERT_TRUE(fixture.render(DXGI_FORMAT_R16G16B16A16_FLOAT,8u,W,H,source.data(),base,base,state,{},rejected,error,W,H,geometry,&bad,nullptr,true,&operands))<<error;
      EXPECT_EQ(rejected,expected);
    }
  }
  EXPECT_TRUE(fixture.private_two_axis_source_unchanged());
}

TEST(HostSbsV2LiveWarpGpuTest, PrivateTwoAxisPinQueuedGuardRemainsCurrent) {
  if (!std::getenv("SUNSHINE_TEST_PRIVATE_TWO_AXIS_ROOT")) GTEST_SKIP()<<"No staged private two-axis program selected";
  namespace guard=models::research_queued_render_guard_v1;namespace v2=models::depth_coordinate_v2;
  constexpr UINT W=128u,H=16u;std::string error;live_v2_warp_fixture_t fixture,flat;
  ASSERT_TRUE(fixture.initialize(error,warp_shader_e::selected_outline,false,true))<<error;
  ASSERT_TRUE(flat.initialize(error,warp_shader_e::flat_main,false,true))<<error;
  ASSERT_TRUE(private_two_axis_files_t::configure(fixture,error))<<error;
  const auto state=make_live_state(true,true);auto ui=make_subtitle_inputs(W,H,state,2.5f/W);
  ui.matte[8u*W+64u]=1u;ui.snapshot[8u*W+64u]={0x3c00u,0x3c00u,0x3c00u,0x3c00u};
  std::vector<rgba16_bits_t> source(W*H,{0x3400u,0x3400u,0x3400u,0x3c00u});source[8u*W+64u]=ui.snapshot[8u*W+64u];
  const std::vector<float> base(W*H,-1.25f/W);auto selected=make_current_outline_inputs(ui);selected.private_two_axis_ui_pin=true;
  const auto render=[&](live_v2_warp_fixture_t &renderer,const current_outline_inputs_t *operands) {
    std::vector<std::byte> pixels;EXPECT_TRUE(renderer.render(DXGI_FORMAT_R16G16B16A16_FLOAT,8u,W,H,source.data(),base,base,state,{},pixels,error,W,H,{},&ui,nullptr,true,operands))<<error;return pixels;
  };
  const auto completed=render(fixture,&selected),identity=render(flat,nullptr);ASSERT_NE(completed,identity);
  const std::uint64_t clock=std::uint64_t(state[v2::gain_last_observation_low])|(std::uint64_t(state[v2::gain_last_observation_high])<<32u);
  const auto expected=guard::make_expected(71ull|(9ull<<32u),1u,clock);
  const auto packet_for=[&](const guard::expected_t &e) {
    std::vector<std::uint32_t> packet(64u);packet[0u]=1u;packet[1u]=1u^0xd1ec15a5u^0x52434f4fu;
    packet[2u]=e.root[0u];packet[3u]=e.root[1u];packet[4u]=e.root[0u]^0xa3756c91u;packet[5u]=e.root[1u]^0x5c8a936eu;
    packet[6u]=0x47524243u;packet[7u]=0x52434f4fu;std::copy_n(packet.begin()+2u,4u,packet.begin()+8u);
    packet[12u]=0x54535152u;packet[13u]=e.root[2u];packet[14u]=e.root[3u];
    packet[20u]=packet[21u]=packet[22u]=packet[26u]=1u;packet[24u]=(W+15u)/16u;packet[25u]=(H+15u)/16u;return packet;
  };
  const auto configure=[&](const guard::expected_t &owner,const std::vector<std::uint32_t> &packet) {
    return fixture.configure_guard_operands(&owner,&packet,error);
  };
  const auto good=packet_for(expected);selected.private_two_axis_ui_pin_queued=true;
  ASSERT_TRUE(configure(expected,good))<<error;EXPECT_EQ(render(fixture,&selected),completed);
  auto bad=good;bad[6u]=0u;ASSERT_TRUE(configure(expected,bad))<<error;const auto bad_draw_packet=fixture.private_guard_srv;
  EXPECT_EQ(render(fixture,&selected),identity);
  ASSERT_TRUE(configure(expected,good))<<error;selected.shared_ui_pin_draw_guard_override=bad_draw_packet.Get();
  EXPECT_EQ(render(fixture,&selected),identity)<<"A good preparation header cannot authorize a revoked queued draw";
  selected.shared_ui_pin_draw_guard_override=nullptr;
  auto stale=expected;stale.observation[0u]^=1u;ASSERT_TRUE(configure(stale,good))<<error;EXPECT_EQ(render(fixture,&selected),identity);
  ASSERT_TRUE(configure(expected,good))<<error;EXPECT_EQ(render(fixture,&selected),completed);
  EXPECT_TRUE(fixture.private_guard_slots_clear());EXPECT_TRUE(fixture.private_two_axis_source_unchanged());
}

// Input is the immutable actual-GPU fixed-mask report already bound to the
// independent layer specification. Its authored context is reused literally;
// neither route rebrands it as a captured or current-producer observation.
TEST(HostSbsV2LiveWarpGpuTest, PrivateTwoAxisPinKnownLayerComparison) {
  const auto *input_env=std::getenv("SUNSHINE_TEST_PRIVATE_TWO_AXIS_KNOWN_REPORT");
  const auto *output_env=std::getenv("SUNSHINE_TEST_PRIVATE_TWO_AXIS_KNOWN_OUTPUT");
  if(!input_env || !*input_env || !output_env || !*output_env)GTEST_SKIP()<<"Explicit frozen fixed-mask report and NEW output required";
  using files=private_two_axis_files_t;using json=nlohmann::json;namespace fs=std::filesystem;
  try {
    const auto input_bytes=files::read(input_env);const auto input=files::document(input_bytes);
    files::require(input.at("not_production_qualified")==true && input.at("temporal_controller_simulated")==false && input.at("cases").size()>0u,"Expected an explicitly authored fixed-mask report");
    const auto reference_binding=input.at("reference_specification");files::require(files::binding(reference_binding.at("path").get<std::string>())==reference_binding,"Independent reference specification changed");
    const auto reference_bytes=files::read(reference_binding.at("path").get<std::string>());const auto reference=files::document(reference_bytes);
    const fs::path output(output_env);files::require(!fs::exists(output)||(fs::is_directory(output)&&fs::is_empty(output)),"Private two-axis output must be new or empty");fs::create_directories(output);
    std::string error;live_v2_warp_fixture_t fixture;ASSERT_TRUE(fixture.initialize(error,warp_shader_e::selected_outline,false,true))<<error;
    ASSERT_TRUE(files::configure(fixture,error))<<error;
    std::array<wchar_t,32768u> executable {};const auto executable_size=GetModuleFileNameW(nullptr,executable.data(),DWORD(executable.size()));files::require(executable_size>0u&&executable_size<executable.size(),"Cannot identify private comparison executable");
    json report={{"schema",1u},{"role","private-two-axis-fixed-plane-known-layer-comparison"},{"not_production_qualified",true},{"temporal_controller_simulated",false},
      {"adapter",fixture.adapter_identity()},{"input_report",files::binding(input_env)},{"reference_specification",reference_binding},{"runtime",input.at("runtime")},
      {"test_executable",files::binding(fs::path(executable.data()))},{"fixture_source",files::binding(fs::path(SUNSHINE_SOURCE_DIR)/"tests/unit/test_host_sbs_v2_live_warp.cpp")},
      {"fixture_include",files::binding(fs::path(SUNSHINE_SOURCE_DIR)/"tests/unit/test_host_sbs_v2_two_axis_private.inc")},
      {"private_shader_root",std::getenv("SUNSHINE_TEST_PRIVATE_TWO_AXIS_ROOT")},{"private_shader_closure",std::getenv("SUNSHINE_TEST_PRIVATE_TWO_AXIS_PIN")},
      {"scope","Same fixed source/Base/state/geometry/target/snapshot/M and frozen authored receipt for row-only and full-height two-axis. Original known layers are reference-only. No current producer or live qualification."},{"cases",json::array()}};
    for(std::size_t index=0u;index<input.at("cases").size();++index) {
      const auto &entry=input.at("cases").at(index);const auto shape=entry.at("source").get<std::array<UINT,2u>>(),field=entry.at("field").get<std::array<UINT,2u>>();const UINT W=shape[0u],H=shape[1u];
      const auto &bindings=entry.at("input_bindings");auto state_values=files::load<std::uint32_t>(entry.at("common_state"),28u);
      models::depth_coordinate_v2::state_words_t state;std::memcpy(state.data(),state_values.data(),sizeof(state));
      host_sbs_v2_geometry_t geometry;const auto geometry_bytes=files::load<std::byte>(entry.at("common_geometry"),sizeof(geometry));std::memcpy(&geometry,geometry_bytes.data(),sizeof(geometry));
      const auto source=files::load<bgra8_t>(bindings.at("source_bgra8"),std::size_t(W)*H),background=files::load<bgra8_t>(bindings.at("background_bgra8"),source.size());
      const auto foreground=files::load<std::array<std::uint8_t,4u>>(bindings.at("foreground_rgba8"),source.size());const auto body=files::load<std::uint8_t>(bindings.at("effective_body_alpha_u8"),source.size());
      const auto base=files::load<float>(bindings.at("base"),std::size_t(field[0u])*field[1u]);
      const fs::path folder=output/("case_"+std::to_string(index));fs::create_directory(folder);
      const auto save=[&](const std::string &name,const void *data,std::size_t size) {const auto path=folder/name;std::ofstream stream(path,std::ios::binary);stream.write(static_cast<const char *>(data),static_cast<std::streamsize>(size));stream.close();files::require(bool(stream),"Cannot save private two-axis artifact");return files::binding(path);};
      const auto save_vector=[&](const std::string &name,const auto &values){return save(name,values.data(),values.size()*sizeof(typename std::decay_t<decltype(values)>::value_type));};
      json result={{"name",entry.at("name")},{"source",shape},{"field",field},{"fixed_plane_bits",entry.at("fixed_plane_bits")},{"input_bindings",bindings},
        {"common_state",entry.at("common_state")},{"common_geometry",entry.at("common_geometry")},{"common_subtitle_constants",entry.at("common_subtitle_constants")},{"routes",json::object()}};
      subtitle_inputs_t reference_ui;
      for(const std::string mask_role:{"control_mask","candidate_mask"}) {
        const auto &context=entry.at("routes").at(mask_role).at("authored_context");subtitle_inputs_t ui;
        files::require(context.at("subtitle_constants")==entry.at("common_subtitle_constants"),"Mask route changed common b3");
        const auto b3=files::load<std::byte>(entry.at("common_subtitle_constants"),sizeof(ui.constants));std::memcpy(&ui.constants,b3.data(),sizeof(ui.constants));
        const auto scp=files::load<std::uint32_t>(context.at("condition"),6u),sct=files::load<std::uint32_t>(context.at("identity"),32u),receipt=files::load<std::uint32_t>(context.at("selected_receipt"),64u);
        std::copy(scp.begin(),scp.end(),ui.condition.begin());std::copy(sct.begin(),sct.end(),ui.identity.begin());ui.width=sct[8u];ui.height=sct[11u];
        files::require(ui.condition[5u]==entry.at("fixed_plane_bits") && ui.constants.color_mode==0u,"Private fixed-plane SDR target must remain unchanged");
        ui.snapshot=files::load<rgba16_bits_t>(bindings.at("snapshot"),std::size_t(ui.width)*ui.height);
        ui.matte=files::load<std::uint32_t>(bindings.at(mask_role=="control_mask"?"control_matte":"candidate_matte"),ui.snapshot.size());
        files::require(std::all_of(ui.matte.begin(),ui.matte.end(),[](auto v){return v<=1u;}),"Private two-axis mask must remain actual literal binary support");
        auto selected=make_current_outline_inputs(ui);selected.matte=ui.matte;std::copy(receipt.begin(),receipt.end(),selected.receipt.begin());reference_ui=ui;
        for(bool two_axis:{false,true}) {
          const std::string name=mask_role+(two_axis?"_two_axis":"_row_only");auto operands=selected;
          operands.production_shared_ui_pin=!two_axis;operands.private_two_axis_ui_pin=two_axis;
          shared_ui_pin_diagnostics_t diagnostic;operands.shared_ui_pin_diagnostic_output=&diagnostic;std::vector<std::byte> pixels;
          ASSERT_TRUE(fixture.render(DXGI_FORMAT_B8G8R8A8_UNORM,4u,W,H,source.data(),base,base,state,{},pixels,error,field[0u],field[1u],geometry,&ui,nullptr,true,&operands))<<name<<": "<<error;
          files::require(diagnostic.selected_receipt_authorized,"Frozen authored receipt must authenticate for both private comparison routes");
          json pin={{"shape",diagnostic.shape},{"field",save_vector(name+".field.f32",diagnostic.field)},{"authorized_weights",save_vector(name+".weights.f32",diagnostic.authorized_weights)},
            {"active_rows",save_vector(name+".rows.u32",diagnostic.active_rows)},{"row_header",save_vector(name+".header.u32",diagnostic.selected_receipt_header)},
            {"selected_receipt_authorized",diagnostic.selected_receipt_authorized},{"active_rows_role",two_axis?"seed-or-modified-row":"seed-row"}};
          if(two_axis) {pin["original_native_Base"]=save_vector(name+".original-Base.f32",diagnostic.private_original_base);pin["horizontal_distance"]=save_vector(name+".horizontal-distance.f32",diagnostic.private_horizontal_distance);pin["final_distance"]=save_vector(name+".final-distance.f32",diagnostic.private_final_distance);pin["logical_resource_bytes"]=diagnostic.private_logical_resource_bytes;}
          result["routes"][name]={{"authored_context",context},{"context_role","frozen-authored-context-not-captured-or-current-producer"},{"color",save_vector(name+".color.bgra8",pixels)},{"shared_warp",std::move(pin)}};
        }
      }
      live_v2_warp_fixture_t::private_layer_output_t layered;
      ASSERT_TRUE(fixture.render_private_layer_reference(W,H,background,foreground,body,base,state,field[0u],field[1u],geometry,reference_ui,reference.at("private_layer_shader_root").get<std::string>(),reference.at("private_layer_source_closure_sha256").get<std::string>(),layered,error))<<error;
      UINT max_error=0u;for(std::size_t at=0u;at<source.size();++at) {const std::array<UINT,3u> expected {source[at].r,source[at].g,source[at].b};for(UINT channel=0u;channel<3u;++channel)max_error=std::max(max_error,UINT(std::abs(int(std::to_integer<unsigned>(layered.reconstructed_rgba8[at*4u+channel]))-int(expected[channel]))));}
      ASSERT_LE(max_error,1u)<<"Original independent layers must reconstruct the common immutable source";
      const auto premultiplied=unpack_rgba32f(layered.shifted_foreground_rgba32f);std::vector<float> alpha;alpha.reserve(premultiplied.size());for(const auto &value:premultiplied)alpha.push_back(value.a);
      result["reference"]={{"clean_base_background",save_vector("reference.clean-base.bgra8",layered.base_background_bgra8)},{"layered_color",save_vector("reference.layered.rgba8",layered.reference_rgba8)},
        {"foreground_alpha",save_vector("reference.alpha.f32",alpha)},{"body_alpha",save_vector("reference.body-alpha.f32",layered.shifted_body_alpha_f32)},{"source_reconstruction",save_vector("reference.source.rgba8",layered.reconstructed_rgba8)},{"source_reconstruction_max_RGB8_error",max_error}};
      report["cases"].push_back(std::move(result));
    }
    files::require(files::read(input_env)==input_bytes&&files::read(reference_binding.at("path").get<std::string>())==reference_bytes&&fixture.private_two_axis_source_unchanged(),"Private two-axis source/input changed during comparison");
    const auto text=report.dump(2);std::ofstream stream(output/"report.json",std::ios::binary);stream<<text;files::require(bool(stream),"Cannot save private two-axis report");
  } catch(const std::exception &failure) {FAIL()<<failure.what();}
}
