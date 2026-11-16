// Independent source layers and alpha truth. No production colour/component verdict
// participates in authoring, and neither alpha nor glyph membership is a GPU input.
void expect_authored_trusted_inner(bool coloured, bool gray_wall, bool shadow,
                                  bool black_outline, bool caption, UINT frame, bool current_support = true,
                                  bool empty_original = false, float current_probability = 1.0f) {
  author(true, frame, 0u, false, true, false, 384u, 64u, 3u);
  const std::size_t count = std::size_t(W_) * H_;
  std::vector<pixel8> scene(count), ink(count, coloured ? pixel8{75u,165u,225u,255u} : pixel8{225u,225u,226u,255u});
  std::vector<std::uint8_t> fill_alpha(count, 0u), shadow_alpha(count, 0u), outline_alpha(count, 0u);
  std::vector<UINT> glyph(count, 0u), layer_owner(count, 0u), counter(count, 0u);
  // Continuous surfaces reach the search boundary. In skin mode the bright stripe
  // occupies the same coarse RGB bin as white but has a higher red value.
  for (UINT y = 0u; y < H_; ++y) for (UINT x = 0u; x < W_; ++x) {
    if (empty_original) scene[y*W_+x] = {248u,248u,248u,255u};
    else if (gray_wall) {
      const auto value = std::uint8_t(188u + (x + 2u*y) / 24u);
      scene[y*W_+x] = {value,value,value,255u};
    } else {
      scene[y*W_+x] = y >= 30u && y < 46u ?
        pixel8{std::uint8_t(232u+(x+2u*y)/48u),std::uint8_t(198u+(2u*x+y)/32u),std::uint8_t(191u+(x+3u*y)/32u),255u} :
        pixel8{std::uint8_t(208u+(x+2u*y)/24u),std::uint8_t(155u+(2u*x+y)/24u),std::uint8_t(115u+(x+y)/16u),255u};
    }
  }
  const auto paint = [&](UINT l, UINT t, UINT r, UINT b, UINT component) {
    for (UINT y = t; y < b; ++y) for (UINT x = l; x < r; ++x) {
      glyph[y*W_+x] = component;
      fill_alpha[y*W_+x] = caption ? 255u : 0u;
    }
  };
  // Three geometric glyphs: a hollow ring, two stems/crossbar, and a cap/stem.
  paint(112u,24u,144u,29u,1u);paint(112u,47u,144u,52u,1u);
  paint(112u,29u,117u,47u,1u);paint(139u,29u,144u,47u,1u);
  paint(170u,24u,175u,52u,2u);paint(197u,24u,202u,52u,2u);paint(175u,35u,197u,40u,2u);
  paint(228u,24u,272u,29u,3u);paint(248u,29u,253u,52u,3u);
  for (UINT y = 29u; y < 47u; ++y) for (UINT x = 117u; x < 139u; ++x) counter[y*W_+x] = 1u;
  if (caption) for (UINT y = 0u; y < H_; ++y) for (UINT x = 0u; x < W_; ++x) {
    UINT shadow_sum = 0u, owner = glyph[y*W_+x];bool outline = false;
    for (int dy = -3; dy <= 3; ++dy) for (int dx = -3; dx <= 3; ++dx) {
      const int sx = int(x) - dx, sy = int(y) - dy;
      if (black_outline && sx >= 0 && sy >= 0 && sx < int(W_) && sy < int(H_) && fill_alpha[sy*W_+sx] != 0u) {
        outline = true;owner = glyph[sy*W_+sx];
      }
      const int bx = int(x) - 3 - dx, by = int(y) - 4 - dy;
      if (shadow && std::abs(dx) <= 2 && std::abs(dy) <= 2 && bx >= 0 && by >= 0 && bx < int(W_) && by < int(H_) && fill_alpha[by*W_+bx] != 0u)
        { shadow_sum += UINT(3-std::abs(dx)) * UINT(3-std::abs(dy));owner = glyph[by*W_+bx]; }
    }
    shadow_alpha[y*W_+x] = std::uint8_t((96u * shadow_sum + 40u) / 81u);
    outline_alpha[y*W_+x] = black_outline && outline ? 255u : 0u;
    layer_owner[y*W_+x] = owner;
  }
  std::vector<UINT> truth(count, 0u);
  const auto over = [](pixel8 bottom, pixel8 top, UINT alpha) {
    for (UINT c = 0u; c < 3u; ++c) bottom[c] = std::uint8_t((alpha*top[c] + (255u-alpha)*bottom[c] + 127u)/255u);
    return bottom;
  };
  origin_pixels_.assign(std::size_t(W_)*fullH_, pixel8{128u,128u,128u,255u});
  for (UINT y = 0u; y < H_; ++y) for (UINT x = 0u; x < W_; ++x) {
    const UINT i = y*W_+x;
    auto value = over(scene[i], pixel8{96u,96u,96u,255u}, shadow_alpha[i]);
    value = over(value, pixel8{0u,0u,0u,255u}, outline_alpha[i]);
    value = over(value, ink[i], fill_alpha[i]);
    origin_pixels_[(y+top_)*W_+x] = value;
    truth[i] = fill_alpha[i] != 0u || shadow_alpha[i] != 0u || outline_alpha[i] != 0u ? 1u : 0u;
  }
  // Fixed broad evidence and broad current boxes, identical for caption/no-caption.
  // A false positive OCR search is intentional in the no-caption controls.
  std::fill(probability_words_.begin(), probability_words_.end(), current_probability);
  std::fill(support_words_.begin(), support_words_.end(), current_support ? 1u : 0u);
  upload_pixels_from_origin();
  auto old_program = program_;
  old_program.trusted_inner_publish_shaders = {};
  old_program.foreground_edge_shaders[0u] = foreground_edge_shaders_[0u].Get();
  auto new_program = old_program;
  new_program.trusted_inner_publish_shaders[0u] = trusted_inner_shaders_[0u].Get();
  ring::resources_t old_owner, new_owner;
  ASSERT_TRUE(old_owner.ensure(device_.Get(), W_, H_, error_)) << error_;
  ASSERT_TRUE(new_owner.ensure(device_.Get(), W_, H_, error_)) << error_;
  ASSERT_TRUE(old_owner.dispatch(context_.Get(), colour_ptrs(), graph_ptrs(), cb_.resource.Get(), rgb_.srv.Get(),
    probability_.srv.Get(), support_.srv.Get(), ocr_.srv.Get(), root_.resource.Get(), 240u, error_, &old_program)) << error_;
  ASSERT_TRUE(new_owner.dispatch(context_.Get(), colour_ptrs(), graph_ptrs(), cb_.resource.Get(), rgb_.srv.Get(),
    probability_.srv.Get(), support_.srv.Get(), ocr_.srv.Get(), root_.resource.Get(), 240u, error_, &new_program)) << error_;
  ASSERT_TRUE(new_owner.trusted_inner_enabled());
  const auto old_e = old_owner.enclosure_views(), new_e = new_owner.enclosure_views();
  EXPECT_EQ(read(old_owner.baseline_views().body.Get()), read(new_owner.baseline_views().body.Get()));
  EXPECT_EQ(read(old_owner.baseline_views().matte.Get()), read(new_owner.baseline_views().matte.Get()));
  EXPECT_EQ(read(old_owner.baseline_views().validation_receipt.Get()), read(new_owner.baseline_views().validation_receipt.Get()));
  EXPECT_EQ(read(old_e.body.Get()), read(new_e.body.Get()));
  EXPECT_EQ(read(old_e.matte.Get()), read(new_e.matte.Get()));
  EXPECT_EQ(read(old_e.receipt.Get()), read(new_e.receipt.Get()));
  EXPECT_EQ(read(old_e.all_body_bits.Get()), read(new_e.all_body_bits.Get()));
  EXPECT_EQ(read(old_e.selected_bits.Get()), read(new_e.selected_bits.Get()));
  EXPECT_EQ(read(old_e.cover_bits.Get()), read(new_e.cover_bits.Get()));
  EXPECT_EQ(read(old_e.family_models.Get()), read(new_e.family_models.Get()));
  EXPECT_EQ(read(old_owner.raw_views().receipt.Get()), read(new_owner.raw_views().receipt.Get()));
  EXPECT_EQ(read(old_owner.raw_views().matte.Get()), read(new_owner.raw_views().matte.Get()));
  if (foreground_edge_shaders_[0u]) {
    EXPECT_EQ(read(old_owner.foreground_edge_original_mask()), read(new_owner.foreground_edge_original_mask()));
  }
  const auto old_m = words(read(old_owner.mutable_views().matte.Get()));
  const auto new_m = words(read(new_owner.mutable_views().matte.Get()));
  const auto trusted = words(read(new_owner.trusted_inner_mask()));
  ASSERT_EQ(old_m.size(), count);ASSERT_EQ(new_m.size(), count);ASSERT_EQ(trusted.size(), count);
  std::size_t added = 0u, added_true = 0u, added_false = 0u, counter_added = 0u;
  std::array<std::size_t,3u> fills {}, old_fills {}, new_fills {};
  std::array<std::size_t,3u> layers {}, old_layers {}, new_layers {};
  for (std::size_t i = 0u; i < count; ++i) {
    ASSERT_LE(old_m[i],1u);ASSERT_LE(new_m[i],1u);
    EXPECT_LE(old_m[i],new_m[i]) << "Lost an existing mask pixel";
    if (new_m[i] != 0u && old_m[i] == 0u) {
      EXPECT_NE(trusted[i],0u) << "Only the trusted inner plane may add a final pixel";
      ++added;if (truth[i] != 0u) ++added_true;else ++added_false;
      if (counter[i] != 0u && truth[i] == 0u) ++counter_added;
    }
    if (fill_alpha[i] == 255u) {
      const auto component = glyph[i]-1u;++fills[component];old_fills[component] += old_m[i];new_fills[component] += new_m[i];
    }
    if (truth[i] != 0u) {
      ASSERT_GE(layer_owner[i],1u);ASSERT_LE(layer_owner[i],3u);
      const auto component = layer_owner[i]-1u;++layers[component];old_layers[component] += old_m[i];new_layers[component] += new_m[i];
    }
  }
  const auto label = std::string(coloured ? "colour" : "white") + (empty_original ? "_flat_bright_empty_body" : gray_wall ? "_gray_bridge" : "_skin_counter") +
    (black_outline ? "_black_outline" : shadow ? "_soft_shadow" : "_no_outline") + (caption ? "_caption" : "_no_caption") + (current_support ? "" : "_no_support") +
    (current_probability==1.0f ? std::string() : "_P"+std::to_string(current_probability));
  nlohmann::json metrics {{"scene",label},{"independent_alpha",true},{"broad_P",current_probability},{"broad_support",current_support ? 1u : 0u},
    {"added",added},{"added_true",added_true},{"added_false",added_false},{"transparent_counter_added",counter_added},
    {"existing_foreground_stage_enabled",bool(foreground_edge_shaders_[0u])},
    {"glyph_opaque_fill_pixels",fills},{"old_glyph_opaque_fill_marked",old_fills},{"new_glyph_opaque_fill_marked",new_fills},
    {"whole_glyph_layer_pixels",layers},{"old_whole_glyph_layer_marked",old_layers},{"new_whole_glyph_layer_marked",new_layers},
    {"semantic_scope","Independent source-alpha truth for these authored scenes; broad P/S does not distinguish a glyph from an identical enclosed scene object"},
    {"performance_claim",false}};
  EXPECT_EQ(counter_added,0u);
  if (empty_original) {
    EXPECT_EQ(words(read(old_owner.baseline_views().body.Get())),std::vector<UINT>(count,0u));
    EXPECT_EQ(old_m,std::vector<UINT>(count,0u));
  }
  if (caption && current_probability>0.2f) {
    EXPECT_GE(100u*added_true,99u*added) << metrics.dump();
    for (UINT component = 0u; component < 3u; ++component) {
      ASSERT_GT(fills[component],0u);EXPECT_GE(100u*new_fills[component],95u*fills[component]) << metrics.dump();
    }
    if (!black_outline) EXPECT_GT(added,0u) << "Fixture must exercise the new route";
  } else EXPECT_EQ(added,0u) << metrics.dump();
  RecordProperty(label,metrics.dump());
  if (const auto *root = std::getenv("SUNSHINE_TEST_AUTHORED_TRUSTED_INNER_OUTPUT")) {
    const auto folder = std::filesystem::path(root)/label;
    ASSERT_FALSE(std::filesystem::exists(folder));ASSERT_TRUE(std::filesystem::create_directories(folder));
    nlohmann::json artifacts = nlohmann::json::object();
    const auto emit = [&](const char *name, const auto &values) {
      const auto bytes = raw(values);std::ofstream f(folder/name,std::ios::binary);f.write(bytes.data(),std::streamsize(bytes.size()));f.close();require(bool(f));
      artifacts[name] = {{"bytes",bytes.size()},{"sha256",hash_bytes(bytes)}};
    };
    emit("scene.rgba8",scene);emit("ink.rgba8",ink);emit("fill_alpha.u8",fill_alpha);emit("shadow_alpha.u8",shadow_alpha);
    emit("outline_alpha.u8",outline_alpha);emit("glyph_component.u32",glyph);emit("layer_component.u32",layer_owner);emit("truth.u32",truth);emit("source_full.rgba8",origin_pixels_);
    emit("P.f32",probability_words_);emit("S.u32",support_words_);emit("OCR.u32",ocr_words_);emit("C96.u32",cb_words_);
    emit("SCT.u32",sct_words_);emit("UIP.u32",uip_words_);emit("root.u32",root_words_);
    emit("old_M.u32",old_m);emit("new_M.u32",new_m);emit("trusted_inner_bits.u32",trusted);
    emit("old_CBC.u32",read(old_e.receipt.Get()));emit("new_CBC.u32",read(new_e.receipt.Get()));
    emit("old_F2.u32",read(old_owner.mutable_views().calibration_data.Get()));
    emit("new_F2.u32",read(new_owner.mutable_views().calibration_data.Get()));
    emit("trusted_publication_arguments.u32",read(new_e.trusted_publication_arguments.Get()));
    metrics["artifacts"] = artifacts;metrics["native_band"] = {W_,H_};metrics["full_source"] = {W_,fullH_,top_};
    std::ofstream f(folder/"report.json");f << metrics.dump(2);f.close();require(bool(f));
  }
}

// Independently authored identical glyphs, translated by an exact native grid
// period, create equal-frequency colour families. Neither alpha nor palette
// metadata is fed back to the producer. A current-family tie must abstain.
void expect_trusted_inner_equal_families() {
  author(true,0x2944u,0u,false,true,false,384u,64u,3u);
  rect(0u,0u,W_,H_,{248u,248u,248u,255u});
  const std::array<pixel8,2u> inks{pixel8{225u,225u,226u,255u},pixel8{75u,165u,225u,255u}};
  for(UINT component=0u;component<2u;++component) {
    const UINT x=112u+96u*component;
    rect(x,24u,x+32u,29u,inks[component]);rect(x,47u,x+32u,52u,inks[component]);
    rect(x,29u,x+5u,47u,inks[component]);rect(x+27u,29u,x+32u,47u,inks[component]);
  }
  std::fill(probability_words_.begin(),probability_words_.end(),1.0f);
  std::fill(support_words_.begin(),support_words_.end(),1u);
  upload_pixels_from_origin();
  // Verify the input's intended tie on the fixed canonical64x8 source grid.
  // This measures authored RGB only, not a producer colour/component verdict.
  std::array<UINT,2u> frequencies{};
  const auto cw=cb_words_[10u]-cb_words_[8u],ch=cb_words_[11u]-cb_words_[9u];
  for(UINT i=0u;i<512u;++i) {
    const UINT x=(((ocr_words_[16u]-cb_words_[8u])*128u+(2u*(i%64u)+1u)*(ocr_words_[18u]-ocr_words_[16u]))*W_)/(128u*cw);
    const UINT y=(((ocr_words_[17u]-cb_words_[9u])*16u+(2u*(i/64u)+1u)*(ocr_words_[19u]-ocr_words_[17u]))*fullH_)/(16u*ch);
    const auto value=origin_pixels_[y*W_+x];
    for(UINT component=0u;component<2u;++component)if(value==inks[component])++frequencies[component];
  }
  ASSERT_GE(frequencies[0u],2u);ASSERT_EQ(frequencies[0u],frequencies[1u]);
  auto off=program_;off.trusted_inner_publish_shaders={};
  off.foreground_edge_shaders[0u]=foreground_edge_shaders_[0u].Get();
  auto on=off;on.trusted_inner_publish_shaders[0u]=trusted_inner_shaders_[0u].Get();
  ring::resources_t old_owner,new_owner;
  ASSERT_TRUE(old_owner.ensure(device_.Get(),W_,H_,error_)) << error_;
  ASSERT_TRUE(new_owner.ensure(device_.Get(),W_,H_,error_)) << error_;
  const auto submit=[&](ring::resources_t &owner,const auto &program) {
    return owner.dispatch(context_.Get(),colour_ptrs(),graph_ptrs(),cb_.resource.Get(),rgb_.srv.Get(),
      probability_.srv.Get(),support_.srv.Get(),ocr_.srv.Get(),root_.resource.Get(),240u,error_,&program);
  };
  ASSERT_TRUE(submit(old_owner,off)) << error_;ASSERT_TRUE(submit(new_owner,on)) << error_;
  ASSERT_TRUE(new_owner.trusted_inner_enabled());
  const std::vector<UINT> empty(W_*H_,0u);
  EXPECT_EQ(words(read(old_owner.mutable_views().matte.Get())),empty);
  EXPECT_EQ(words(read(new_owner.mutable_views().matte.Get())),empty);
  EXPECT_EQ(words(read(new_owner.trusted_inner_mask())),empty);
  EXPECT_EQ(read(old_owner.mutable_views().calibration_data.Get()),read(new_owner.mutable_views().calibration_data.Get()));
  RecordProperty("trusted_inner_equal_family_scope",nlohmann::json{{"independent_source",true},
    {"white_grid_frequency",frequencies[0u]},{"blue_grid_frequency",frequencies[1u]},
    {"broad_P",1u},{"broad_S",1u},{"expected","Unique maximum absent: no TIR or final additions"}}.dump());
}

// Real producer inputs, one retained owner, and fresh-owner controls. No receipt,
// component or family record is injected to make the optional publisher ready.
void expect_trusted_inner_current_lifecycle() {
  const auto author_frame = [&](UINT frame, UINT shift) {
    author(true,frame,0u,false,true,false,384u,64u,3u);
    rect(0u,0u,W_,H_,{248u,248u,248u,255u});
    for (const UINT left : {100u,164u,228u}) {
      rect(left+shift,24u,left+shift+32u,29u,{225u,225u,226u,255u});
      rect(left+shift,47u,left+shift+32u,52u,{225u,225u,226u,255u});
      rect(left+shift,29u,left+shift+5u,47u,{225u,225u,226u,255u});
      rect(left+shift+27u,29u,left+shift+32u,47u,{225u,225u,226u,255u});
    }
    std::fill(probability_words_.begin(),probability_words_.end(),1.0f);
    std::fill(support_words_.begin(),support_words_.end(),1u);
    upload_pixels_from_origin();
  };
  const auto submit = [&](ring::resources_t &owner,ID3D11ComputeShader *publisher) {
    auto program=program_;
    program.foreground_edge_shaders[0u]=foreground_edge_shaders_[0u].Get();
    program.trusted_inner_publish_shaders[0u]=publisher;
    return owner.dispatch(context_.Get(),colour_ptrs(),graph_ptrs(),cb_.resource.Get(),rgb_.srv.Get(),
      probability_.srv.Get(),support_.srv.Get(),ocr_.srv.Get(),root_.resource.Get(),240u,error_,&program);
  };
  const auto snapshot = [&](ring::resources_t &owner) {
    return std::map<std::string,std::vector<char>> {
      {"M",read(owner.mutable_views().matte.Get())},
      {"F2",read(owner.mutable_views().calibration_data.Get())},
      {"BODY",read(owner.baseline_views().body.Get())},
      {"CBC",read(owner.enclosure_views().receipt.Get())},
      {"Raw",read(owner.raw_views().receipt.Get())},
      {"RawM",read(owner.raw_views().matte.Get())}};
  };
  const auto expect_accounting = [&](const auto &observation,bool recovered) {
    const auto mask=words(observation.at("M")),data=words(observation.at("F2"));
    const auto count=UINT(std::count(mask.begin(),mask.end(),1u));
    EXPECT_EQ(words(observation.at("BODY")),std::vector<UINT>(W_*H_,0u));
    EXPECT_EQ(words(observation.at("RawM")),std::vector<UINT>(W_*H_,0u));
    ASSERT_EQ(data.size(),68608u);EXPECT_EQ(data[7u],count);EXPECT_EQ(data[70u],count);
    EXPECT_EQ(data[73u],0u);EXPECT_EQ(data[76u],0u);
    if(recovered)EXPECT_GT(count,0u);else EXPECT_EQ(count,0u);
  };
  ring::resources_t owner;
  author_frame(0x2960u,0u);
  ASSERT_TRUE(owner.ensure(device_.Get(),W_,H_,error_)) << error_;
  ASSERT_TRUE(submit(owner,trusted_inner_shaders_[0u].Get())) << error_;
  ASSERT_TRUE(owner.trusted_inner_enabled());
  const auto initial=snapshot(owner);
  const auto initial_bits=read(owner.trusted_inner_mask());
  const auto initial_bit_words=words(initial_bits);
  ASSERT_NO_FATAL_FAILURE(expect_accounting(initial,true));
  ASSERT_GT(std::count(initial_bit_words.begin(),initial_bit_words.end(),1u),0);
  author_frame(0x2961u,12u);root_words_[60u]=0u;upload();
  ASSERT_TRUE(submit(owner,trusted_inner_shaders_[0u].Get())) << error_;
  EXPECT_EQ(snapshot(owner),initial);EXPECT_EQ(read(owner.trusted_inner_mask()),initial_bits);
  author_frame(0x2962u,12u);
  ASSERT_TRUE(submit(owner,trusted_inner_shaders_[0u].Get())) << error_;
  const auto refreshed=snapshot(owner);
  EXPECT_NE(refreshed.at("M"),initial.at("M"));
  ASSERT_NO_FATAL_FAILURE(expect_accounting(refreshed,true));
  ring::resources_t fresh_enabled;
  ASSERT_TRUE(fresh_enabled.ensure(device_.Get(),W_,H_,error_)) << error_;
  ASSERT_TRUE(submit(fresh_enabled,trusted_inner_shaders_[0u].Get())) << error_;
  EXPECT_EQ(refreshed,snapshot(fresh_enabled));
  EXPECT_EQ(read(owner.trusted_inner_mask()),read(fresh_enabled.trusted_inner_mask()));
  author_frame(0x2963u,24u);
  ASSERT_TRUE(submit(owner,nullptr)) << error_;
  ASSERT_FALSE(owner.trusted_inner_enabled());EXPECT_EQ(owner.trusted_inner_mask(),nullptr);
  const auto disabled=snapshot(owner);
  ASSERT_NO_FATAL_FAILURE(expect_accounting(disabled,false));
  ring::resources_t fresh_disabled;
  ASSERT_TRUE(fresh_disabled.ensure(device_.Get(),W_,H_,error_)) << error_;
  ASSERT_TRUE(submit(fresh_disabled,nullptr)) << error_;
  EXPECT_EQ(disabled,snapshot(fresh_disabled));
  ComPtr<ID3D11Device> foreign;ComPtr<ID3D11DeviceContext> foreign_context;
  const D3D_FEATURE_LEVEL level=D3D_FEATURE_LEVEL_11_0;D3D_FEATURE_LEVEL actual{};
  ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_WARP,nullptr,0u,&level,1u,
    D3D11_SDK_VERSION,&foreign,&actual,&foreign_context)));
  const auto &spec=models::current_outline_runtime::trusted_inner_specs;
  const auto source=cache::snapshot_sources(std::getenv("SUNSHINE_TEST_TRUSTED_INNER_SHADER_ROOT"),spec);
  ASSERT_EQ(cache::source_closure_sha256(source),std::getenv("SUNSHINE_TEST_TRUSTED_INNER_SHADER_PIN"));
  const auto code=cache::get(source,spec[0u]);ASSERT_TRUE(code&&!code->empty());
  ComPtr<ID3D11ComputeShader> wrong;
  ASSERT_TRUE(SUCCEEDED(foreign->CreateComputeShader(code->data(),code->size(),nullptr,&wrong)));
  ASSERT_TRUE(submit(owner,wrong.Get())) << error_;
  EXPECT_FALSE(owner.trusted_inner_enabled());EXPECT_EQ(owner.trusted_inner_mask(),nullptr);
  EXPECT_EQ(snapshot(owner),disabled);
  RecordProperty("trusted_inner_lifecycle_scope",
    "Real producer current frames; enabled/Root0/moved-enabled/disabled one-owner sequence; fresh controls, "
    "literal original BODY=0 and RawM=0, final M/unique-add agreement, missing/foreign publisher old-route equality");
}
