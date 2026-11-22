TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, InsideTopologyResearchReplaysAllUnitsAndPreservesCurrentAa) {
  if (!body_witness_research_configured()) GTEST_SKIP() << "Explicit private topology repository root required";
  author_band(5u, 9900u);rect(79u, 60u, 80u, 61u, {128u, 128u, 128u, 255u});rect(79u, 62u, 80u, 63u, {64u, 64u, 64u, 255u});
  ocr_words_[3u] = ocr_words_[4u] = 2u;uip_words_[3u] = 3u;
  std::copy_n(ocr_words_.begin() + 16u, 4u, ocr_words_.begin() + 24u);std::copy_n(ocr_words_.begin() + 144u, 4u, ocr_words_.begin() + 152u);
  upload_pixels_from_origin();ASSERT_TRUE(run()) << error_;const auto original = words(outline_matte());
  const auto [topology, filtered] = probe_current_inside_topology();
  export_inside_topology_probe("all_units_aa", topology, filtered, "Actual two-unit canonical replay and authored gray/dark inner-AA; no CPU connectivity or regenerated alpha");
  ASSERT_TRUE(topology.byteexact);ASSERT_EQ(topology.header_words.size(), 8u);
  EXPECT_EQ(topology.header_words[0u], 1u);EXPECT_EQ(topology.header_words[1u], 1u);
  for (UINT unit = 2u; unit < 8u; ++unit) EXPECT_EQ(topology.header_words[unit], 0u);
  EXPECT_EQ(filtered.filtered, original);EXPECT_EQ(filtered.filtered[60u * W_ + 79u], 1u);EXPECT_EQ(filtered.filtered[62u * W_ + 79u], 1u);
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, InsideTopologyResearchPreservesExteriorAaAndRetainsOldDiagnostic) {
  if (!body_witness_research_configured()) GTEST_SKIP() << "Explicit private topology repository root required";
  author_band(5u, 9901u);rect(79u, 60u, 80u, 61u, {160u, 160u, 160u, 255u});
  rect(95u, 86u, 96u, 113u, {255u, 160u, 255u, 255u});upload_pixels_from_origin();ASSERT_TRUE(run()) << error_;
  const auto control = words(outline_matte());const UINT q = 60u * W_ + 79u;
  const auto old = probe_body_witness_current();const auto [topology, filtered] = probe_current_inside_topology();
  export_inside_topology_probe("exterior_aa", topology, filtered, "Authored current nonglobal-family AA pixel in first glyph; nonwhite physical opening preserves colour BODY but makes Inside graph exterior");
  ASSERT_TRUE(topology.byteexact);ASSERT_EQ(words(body())[q], 1u);ASSERT_EQ(control[q], 1u);
  const auto exterior = words(topology.artifacts.at("exterior.u32"));const auto closed = words(topology.artifacts.at("closed_unseeded.u32"));
  ASSERT_NE(exterior[q] & 1u, 0u);ASSERT_EQ(closed[q] & 1u, 0u);
  EXPECT_NE(old.rejected[q], 0u) << "The earlier diagnostic remains a separately observable rejected policy";
  EXPECT_EQ(filtered.rejected[q], 0u);EXPECT_EQ(filtered.filtered[q], 1u);EXPECT_NE(filtered.reasons[q] & 2048u, 0u);
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, InsideTopologyResearchObservesClosedUnseededSceneWithoutTouchingSeededGlyphs) {
  if (!body_witness_research_configured()) GTEST_SKIP() << "Explicit private topology repository root required";
  const UINT q = author_topology_scene_counter(9902u, false);ASSERT_TRUE(run()) << error_;
  const auto control = words(outline_matte());const auto [topology, filtered] = probe_current_inside_topology();
  export_inside_topology_probe("closed_scene_counter", topology, filtered, "Independent authored noncaption scene ring/counter, sourceRGB160 outside original learning core; actual canonical graph decides closed/unseeded versus glyph seed");
  ASSERT_TRUE(topology.byteexact);const auto closed = words(topology.artifacts.at("closed_unseeded.u32"));
  ASSERT_NE(closed[q] & 1u, 0u);ASSERT_EQ(words(read(helper_.mutable_views().inside_bits.Get()))[q] & 1u, 0u);
  ASSERT_EQ(words(body())[q], 1u);ASSERT_EQ(control[q], 1u);
  EXPECT_NE(filtered.rejected[q] & 1u, 0u);EXPECT_EQ(filtered.filtered[q], 0u);EXPECT_NE(filtered.reasons[q] & 8192u, 0u);
  const UINT glyph = 60u * W_ + 90u;ASSERT_EQ(control[glyph], 1u);EXPECT_EQ(filtered.filtered[glyph], 1u);
  RecordProperty("closed_scene_counter_truth", "Authored scene object is alpha-zero by construction; this does not label genuine character counters or unlabelled video");
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, InsideTopologyResearchInvalidBoundaryCannotCreateErasureCertificate) {
  if (!body_witness_research_configured()) GTEST_SKIP() << "Explicit private topology repository root required";
  const UINT q = author_topology_scene_counter(9903u, true);ASSERT_TRUE(run()) << error_;
  const auto baseline = words(outline_matte());const auto source = read(rgb_.resource.Get());
  auto copied = texture(W_, H_, DXGI_FORMAT_R16G16B16A16_FLOAT);context_->UpdateSubresource(copied.resource.Get(), 0u, nullptr, source.data(), W_ * 8u, 0u);
  // Explicit invalid current-source wall cuts the two-pixel opening. Canonical
  // Inside treats it as nonpassable; the additional root-bad reduction must
  // make the touched root UNKNOWN rather than inventing a closed certificate.
  const std::array<pixel16, 2u> invalid {{{0u, 0u, 0u, 0u}, {0u, 0u, 0u, 0u}}};
  const D3D11_BOX wall {73u, 108u, 0u, 75u, 109u, 1u};context_->UpdateSubresource(copied.resource.Get(), 0u, &wall, invalid.data(), 16u, 0u);
  const auto [topology, filtered] = probe_current_inside_topology(copied.srv.Get());
  export_inside_topology_probe("invalid_scene_wall", topology, filtered, "Kernel-only COPIED source inserts invalid wall; actual producer remains unchanged. Invalid boundary may not certify closure even when completed Inside bits still match.");
  ASSERT_TRUE(topology.byteexact);const auto unknown = words(topology.artifacts.at("unknown.u32"));const auto closed = words(topology.artifacts.at("closed_unseeded.u32"));
  EXPECT_NE(unknown[q] & 1u, 0u);EXPECT_EQ(closed[q] & 1u, 0u);EXPECT_EQ(filtered.rejected[q], 0u);EXPECT_EQ(filtered.filtered[q], baseline[q]);
  EXPECT_EQ(read(rgb_.resource.Get()), source);EXPECT_EQ(words(outline_matte()), baseline);
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, InsideTopologyResearchCannotReuseSourceATopologyOnSourceB) {
  if (!body_witness_research_configured()) GTEST_SKIP() << "Explicit private topology repository root required";
  author_band(5u, 9910u);ASSERT_TRUE(run()) << error_;
  const auto [topology_a, accepted_a] = probe_current_inside_topology();
  export_inside_topology_probe("same_source_a", topology_a, accepted_a, "Actual current source A; exact consumed-input join required in addition to replayed InsideBits parity");
  ASSERT_TRUE(topology_a.byteexact);ASSERT_TRUE(accepted_a.metadata.at("topology_current_tuple_usable").get<bool>());
  author_band(5u, 9911u);ASSERT_TRUE(run()) << error_;const auto original_b = words(outline_matte());
  const auto refused = probe_body_with_topology(cb_.resource.Get(), W_, H_, current_body_witness_inputs(), topology_a);
  export_inside_topology_probe("source_b_refuses_topology_a", topology_a, refused, "Actual separately authenticated source B keeps its literal current M; source A's cached valid topology carries no B permission");
  EXPECT_FALSE(refused.metadata.at("topology_consumed_inputs_match").get<bool>());
  EXPECT_FALSE(refused.metadata.at("filter_dispatched").get<bool>());EXPECT_EQ(refused.filtered, original_b);
  EXPECT_TRUE(std::all_of(refused.rejected.begin(), refused.rejected.end(), [](UINT word) { return word == 0u; }));
  const auto gate = words(refused.consumed.at("TopologyGate.u32"));ASSERT_EQ(gate.size(), 4u);EXPECT_EQ(gate[2u], 0u);
  // Same tuple annotations are insufficient too: a copied, otherwise valid
  // RGB override must match the replay's exact consumed source bytes.
  const auto [topology_b, accepted_b] = probe_current_inside_topology();ASSERT_TRUE(topology_b.byteexact);
  const auto source = read(rgb_.resource.Get());auto copy = texture(W_, H_, DXGI_FORMAT_R16G16B16A16_FLOAT);
  context_->UpdateSubresource(copy.resource.Get(), 0u, nullptr, source.data(), W_ * 8u, 0u);
  const pixel16 changed {code_half(129u), code_half(128u), code_half(128u), 0x3c00u};
  const D3D11_BOX pixel {40u, 30u, 0u, 41u, 31u, 1u};context_->UpdateSubresource(copy.resource.Get(), 0u, &pixel, changed.data(), 8u, 0u);
  const auto overridden = probe_body_with_topology(cb_.resource.Get(), W_, H_, current_body_witness_inputs(copy.srv.Get()), topology_b);
  export_inside_topology_probe("copied_rgb_refuses_topology_b", topology_b, overridden, "Kernel-only valid copied RGB changes one source code while C24/SCT remain identical; exact source byte join refuses the old topology");
  EXPECT_FALSE(overridden.metadata.at("topology_consumed_inputs_match").get<bool>());EXPECT_FALSE(overridden.metadata.at("filter_dispatched").get<bool>());
  EXPECT_EQ(overridden.filtered, original_b);EXPECT_TRUE(std::all_of(overridden.rejected.begin(), overridden.rejected.end(), [](UINT word) { return word == 0u; }));
}
