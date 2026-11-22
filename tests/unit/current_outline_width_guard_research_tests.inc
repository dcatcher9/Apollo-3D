TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, WidthGuardResearchKeepsOriginalFullAndThinUnseededAccents) {
  if (!body_witness_research_configured()) GTEST_SKIP() << "Explicit private measured-width repository root required";
  for (const UINT width : {8u, 2u}) {
    const UINT q = author_topology_scene_counter(9940u + width, false);
    if (width == 2u) {rect(70u, 101u, 78u, 107u, {0u, 0u, 0u, 255u});rect(73u, 101u, 75u, 107u, {160u, 160u, 160u, 255u});upload_pixels_from_origin();}
    ASSERT_TRUE(run()) << error_;const auto literal = words(outline_matte());auto [topology, filtered] = probe_width_guard_inputs(current_body_witness_inputs());
    attach_physical_k_authored_alpha(topology, filtered, true);filtered.metadata["independent_authored_truth"]["gray_fill_width"] = width;
    export_width_guard_probe(width == 2u ? "width_guard_thin_accent" : "width_guard_full_accent", topology, filtered, "Genuine detached glyph/diacritic with no exact FG seed; source-identical scene interpretation remains, but measured original K width must preserve visible ink");
    ASSERT_TRUE(topology.byteexact);ASSERT_TRUE(filtered.metadata.at("filter_dispatched").get<bool>());ASSERT_EQ(calibration()[139u], 5u);
    ASSERT_EQ(words(body())[q], 1u);ASSERT_EQ(literal[q], 1u);ASSERT_EQ(words(read(helper_.mutable_views().inside_bits.Get()))[q] & 1u, 0u);
    ASSERT_NE(words(topology.artifacts.at("closed_unseeded.u32"))[q] & 1u, 0u);ASSERT_EQ(words(topology.artifacts.at("seeded.u32"))[q] & 1u, 0u);
    ASSERT_NE(filtered.reasons[q] & 8192u, 0u) << "Actual closed physical proposer must be active";
    ASSERT_NE(filtered.reasons[q] & 32768u, 0u) << "Actual strong local width must invoke the K keep collar";
    EXPECT_EQ(filtered.rejected[q], 0u);EXPECT_EQ(filtered.filtered[q], 1u);EXPECT_EQ(filtered.filtered, literal);
  }
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, WidthGuardResearchKeepsSeededGlyphAndCounterBoundaryAa) {
  if (!body_witness_research_configured()) GTEST_SKIP() << "Explicit private measured-width repository root required";
  author_band(5u, 9950u);rect(79u, 60u, 80u, 61u, {128u, 128u, 128u, 255u});rect(79u, 62u, 80u, 63u, {64u, 64u, 64u, 255u});
  // Independent original glyph-counter layers: source212 is clean counter
  // scene, black is original inner outline, and source160 is original alpha63
  // black AA over that same212 scene (integer source composition rounds160).
  rect(90u, 60u, 102u, 78u, {0u, 0u, 0u, 255u});rect(92u, 62u, 100u, 76u, {212u, 212u, 212u, 255u});rect(92u, 65u, 93u, 66u, {160u, 160u, 160u, 255u});
  ocr_words_[3u] = ocr_words_[4u] = 2u;uip_words_[3u] = 3u;std::copy_n(ocr_words_.begin() + 16u, 4u, ocr_words_.begin() + 24u);std::copy_n(ocr_words_.begin() + 144u, 4u, ocr_words_.begin() + 152u);
  upload_pixels_from_origin();ASSERT_TRUE(run()) << error_;const auto literal = words(outline_matte());auto [topology, filtered] = probe_width_guard_inputs(current_body_witness_inputs());
  attach_width_guard_scene_truth(topology, filtered, 92u, 65u);auto alpha = std::vector<std::uint8_t>(std::size_t(W_) * H_, 0u);auto clean = origin_pixels_;
  for (const auto rect : {std::array<UINT, 4u> {74u, 43u, 116u, 93u}, std::array<UINT, 4u> {155u, 43u, 213u, 93u}}) for (UINT y = rect[1u]; y < rect[3u]; ++y) for (UINT x = rect[0u]; x < rect[2u]; ++x) {alpha[y * W_ + x] = 255u;clean[y * W_ + x] = {128u, 128u, 128u, 255u};}
  for (UINT y = 60u; y < 78u; ++y) for (UINT x = 90u; x < 102u; ++x) clean[y * W_ + x] = {212u, 212u, 212u, 255u};
  for (UINT y = 62u; y < 76u; ++y) for (UINT x = 92u; x < 100u; ++x) alpha[y * W_ + x] = 0u;
  const UINT aa = 65u * W_ + 92u;alpha[aa] = 63u;topology.artifacts["authored_alpha.u8"] = raw(alpha);topology.artifacts["authored_clean_scene.rgba8"] = raw(clean);
  filtered.metadata["independent_authored_truth"]["target_alpha"] = 63u;filtered.metadata["independent_authored_truth"]["target_source_clean_max_RGB8_delta"] = 52u;
  export_width_guard_probe("width_guard_seeded_counter_aa", topology, filtered, "Two actual current units; independently authored gray/dark fill and alpha63 original inner-counter AA over known clean scene; retain original M");
  ASSERT_TRUE(topology.byteexact);ASSERT_EQ(topology.header_words[0u], 1u);ASSERT_EQ(topology.header_words[1u], 1u);
  for (const UINT q : {60u * W_ + 79u, 62u * W_ + 79u, 60u * W_ + 90u, aa}) {ASSERT_EQ(literal[q], 1u);EXPECT_EQ(filtered.filtered[q], 1u);EXPECT_EQ(filtered.rejected[q], 0u);}
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, WidthGuardResearchRemovesOnlyActiveDeepCounterSceneBeyondMeasuredWidth) {
  if (!body_witness_research_configured()) GTEST_SKIP() << "Explicit private measured-width repository root required";
  author_band(5u, 9951u);rect(38u, 48u, 62u, 82u, {0u, 0u, 0u, 255u});rect(42u, 52u, 58u, 78u, {160u, 160u, 160u, 255u});upload_pixels_from_origin();
  ASSERT_TRUE(run()) << error_;const UINT q = 65u * W_ + 50u;const auto literal = words(outline_matte());auto [topology, filtered] = probe_width_guard_inputs(current_body_witness_inputs());
  attach_width_guard_scene_truth(topology, filtered, 50u, 65u);filtered.metadata["independent_authored_truth"]["authored_min_physical_K_Chebyshev_distance"] = 8u;
  export_width_guard_probe("width_guard_deep_scene_counter", topology, filtered, "Authored scene ring outside OCR colour-learning core; target has native distance8 from exact source K versus actual measured width5. Actual BODY/M/closed/unseeded/strong-width prerequisites are mandatory.");
  ASSERT_TRUE(topology.byteexact);ASSERT_EQ(calibration()[139u], 5u);ASSERT_EQ(calibration()[141u], 1u);ASSERT_EQ(words(body())[q], 1u);ASSERT_EQ(literal[q], 1u);
  ASSERT_NE(words(topology.artifacts.at("closed_unseeded.u32"))[q] & 1u, 0u);ASSERT_EQ(words(topology.artifacts.at("seeded.u32"))[q] & 1u, 0u);ASSERT_EQ(words(topology.artifacts.at("unknown.u32"))[q] & 1u, 0u);
  ASSERT_NE(filtered.reasons[q] & 8192u, 0u);ASSERT_EQ(filtered.reasons[q] & (16384u | 32768u | 65536u), 0u);
  EXPECT_NE(filtered.rejected[q] & 1u, 0u);EXPECT_EQ(filtered.filtered[q], 0u);EXPECT_EQ(filtered.filtered[60u * W_ + 90u], 1u);
  EXPECT_EQ(filtered.metadata.at("mask_delta").at("actual_M_additions"), 0u);
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, WidthGuardResearchAbstainsOnMissingOrWeakLocalMeasurement) {
  if (!body_witness_research_configured()) GTEST_SKIP() << "Explicit private measured-width repository root required";
  const UINT q = author_topology_scene_counter(9952u, false);ASSERT_TRUE(run()) << error_;const auto literal = words(outline_matte());const auto original_f2 = calibration();
  const auto [control_topology, control] = probe_width_guard_inputs(current_body_witness_inputs());ASSERT_TRUE(control_topology.byteexact);ASSERT_NE(control.reasons[q] & 32768u, 0u);
  for (const bool weak : {false, true}) {
    auto copied = original_f2;if (weak) copied[152u] = 1u;else {copied[141u] = 0u;copied[139u] = 0u;}
    auto f2 = topology_words(68608u, true);context_->UpdateSubresource(f2.resource.Get(), 0u, nullptr, copied.data(), 0u, 0u);auto inputs = current_body_witness_inputs();inputs[5u] = f2.srv.Get();
    auto [topology, filtered] = probe_width_guard_inputs(inputs);filtered.metadata["operand_override_scope"] = "Kernel-only copied F2 local-width certificate degraded; no production F2 or measured source records changed. Exact canonical/replay/filter consumption joins still mandatory.";
    export_width_guard_probe(weak ? "width_guard_weak_local_width" : "width_guard_missing_local_width", topology, filtered, "Active closed physical proposer must preserve literal M when its own current local width certificate is weak/unavailable; no shared donor substitution");
    ASSERT_TRUE(topology.byteexact);ASSERT_NE(filtered.reasons[q] & 8192u, 0u);ASSERT_NE(filtered.reasons[q] & 16384u, 0u);ASSERT_EQ(literal[q], 1u);
    EXPECT_EQ(filtered.rejected[q], 0u);EXPECT_EQ(filtered.filtered, literal);EXPECT_EQ(calibration(), original_f2);
  }
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, WidthGuardResearchAbstainsOnClippedOrInvalidMeasuredNeighborhood) {
  if (!body_witness_research_configured()) GTEST_SKIP() << "Explicit private measured-width repository root required";
  const UINT q = author_topology_scene_counter(9953u, false);rect(70u, 101u, 72u, 107u, {0u, 0u, 0u, 255u});upload_pixels_from_origin();ASSERT_TRUE(run()) << error_;
  const auto literal = words(outline_matte());const auto source = read(rgb_.resource.Get());const auto original_cbc = receipt();ASSERT_EQ(calibration()[139u], 5u);
  for (const bool invalid_source : {false, true}) {
    auto inputs = current_body_witness_inputs();auto cbc = buffer(512u * 4u);auto copied_rgb = texture(W_, H_, DXGI_FORMAT_R16G16B16A16_FLOAT);
    if (invalid_source) {
      context_->UpdateSubresource(copied_rgb.resource.Get(), 0u, nullptr, source.data(), W_ * 8u, 0u);const pixel16 invalid {0u, 0u, 0u, 0u};
      const D3D11_BOX corner {69u, 98u, 0u, 70u, 99u, 1u};context_->UpdateSubresource(copied_rgb.resource.Get(), 0u, &corner, invalid.data(), 8u, 0u);inputs[0u] = copied_rgb.srv.Get();
    } else {
      auto clipped = original_cbc;clipped[152u] = 70u;context_->UpdateSubresource(cbc.resource.Get(), 0u, nullptr, clipped.data(), 0u, 0u);inputs[4u] = cbc.srv.Get();
    }
    auto [topology, filtered] = probe_width_guard_inputs(inputs);filtered.metadata["operand_override_scope"] = invalid_source ? "Kernel-only copied RGB has invalid sample at first width-neighborhood corner; production source unchanged" : "Kernel-only copied CBC ROI clips the measured neighborhood while retaining actual seeded glyphs and the target's physical closure; production receipt unchanged";
    export_width_guard_probe(invalid_source ? "width_guard_invalid_neighborhood" : "width_guard_clipped_neighborhood", topology, filtered, "Actual closed physical proposer and strong local width remain active; unavailable width-neighborhood evidence must withdraw erasure and retain literal M");
    ASSERT_TRUE(topology.byteexact);ASSERT_EQ(literal[q], 1u);ASSERT_NE(words(topology.artifacts.at("closed_unseeded.u32"))[q] & 1u, 0u);ASSERT_EQ(words(topology.artifacts.at("unknown.u32"))[q] & 1u, 0u);
    ASSERT_NE(filtered.reasons[q] & 8192u, 0u);ASSERT_NE(filtered.reasons[q] & 65536u, 0u);EXPECT_EQ(filtered.rejected[q], 0u);EXPECT_EQ(filtered.filtered[q], 1u);
    EXPECT_EQ(receipt(), original_cbc);EXPECT_EQ(read(rgb_.resource.Get()), source);EXPECT_EQ(words(outline_matte()), literal);
  }
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, WidthGuardResearchRetainsExactSourceAndCanonicalConsumptionGates) {
  if (!body_witness_research_configured()) GTEST_SKIP() << "Explicit private measured-width repository root required";
  author_band(5u, 9954u);ASSERT_TRUE(run()) << error_;const auto [topology_a, filtered_a] = probe_width_guard_inputs(current_body_witness_inputs());ASSERT_TRUE(topology_a.byteexact);ASSERT_TRUE(filtered_a.metadata.at("filter_dispatched").get<bool>());
  author_band(5u, 9955u);ASSERT_TRUE(run()) << error_;const auto literal = words(outline_matte());auto refused = probe_body_with_topology(cb_.resource.Get(), W_, H_, current_body_witness_inputs(), topology_a, true);auto retained_a = topology_a;
  export_width_guard_probe("width_guard_source_b_refuses_a", retained_a, refused, "Actual separately authenticated source B retains literal M when offered source A's cached topology");
  EXPECT_FALSE(refused.metadata.at("filter_dispatched").get<bool>());EXPECT_FALSE(refused.metadata.at("topology_consumed_inputs_match").get<bool>());EXPECT_EQ(refused.filtered, literal);EXPECT_EQ(words(refused.consumed.at("TopologyGate.u32"))[2u], 0u);
  EXPECT_TRUE(std::all_of(refused.rejected.begin(), refused.rejected.end(), [](UINT v) { return v == 0u; }));
  const auto e = helper_.enclosure_views();const auto r = helper_.raw_views();const auto inputs = current_body_witness_inputs();const auto canonical = replay_inside_topology(cb_.resource.Get(), W_, H_, inputs, e.all_body_bits.Get(), r.matte.Get(), e.matte.Get(), program_.inside_shaders, program_.roi_graph);ASSERT_TRUE(canonical.byteexact);
  const auto source = read(rgb_.resource.Get());auto rgb = texture(W_, H_, DXGI_FORMAT_R16G16B16A16_FLOAT);context_->UpdateSubresource(rgb.resource.Get(), 0u, nullptr, source.data(), W_ * 8u, 0u);
  const pixel16 changed {code_half(129u), code_half(128u), code_half(128u), 0x3c00u};const D3D11_BOX pixel {40u, 30u, 0u, 41u, 31u, 1u};context_->UpdateSubresource(rgb.resource.Get(), 0u, &pixel, changed.data(), 8u, 0u);
  const auto changed_inputs = current_body_witness_inputs(rgb.srv.Get());auto mismatched = replay_physical_k_topology(cb_.resource.Get(), W_, H_, changed_inputs, canonical, program_.roi_graph);auto failed = probe_body_with_topology(cb_.resource.Get(), W_, H_, changed_inputs, mismatched, true);
  export_width_guard_probe("width_guard_changed_rgb_refuses_canonical", mismatched, failed, "Kernel-only copied RGB changes one code without matching canonical consumed source; Gate0 and literal M remain mandatory");
  EXPECT_FALSE(mismatched.byteexact);EXPECT_FALSE(failed.metadata.at("filter_dispatched").get<bool>());EXPECT_EQ(failed.filtered, literal);EXPECT_EQ(words(failed.consumed.at("TopologyGate.u32"))[2u], 0u);
  EXPECT_TRUE(std::all_of(failed.rejected.begin(), failed.rejected.end(), [](UINT v) { return v == 0u; }));
}
