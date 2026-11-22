TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, PhysicalKTopologyResearchKeepsSeededGlyphAaAcrossAllUnits) {
  if (!body_witness_research_configured()) GTEST_SKIP() << "Explicit private topology repository root required";
  author_band(5u, 9920u);rect(79u, 60u, 80u, 61u, {128u, 128u, 128u, 255u});rect(79u, 62u, 80u, 63u, {64u, 64u, 64u, 255u});
  // Genuine authored counter inside the first white glyph; no alpha is inferred
  // from the detector. Its original RGB and current production mask are controls.
  rect(92u, 65u, 98u, 71u, {0u, 0u, 0u, 255u});
  ocr_words_[3u] = ocr_words_[4u] = 2u;uip_words_[3u] = 3u;
  std::copy_n(ocr_words_.begin() + 16u, 4u, ocr_words_.begin() + 24u);std::copy_n(ocr_words_.begin() + 144u, 4u, ocr_words_.begin() + 152u);
  upload_pixels_from_origin();ASSERT_TRUE(run()) << error_;const auto control = words(outline_matte());
  const auto [topology, filtered] = probe_current_physical_k_topology();
  export_inside_topology_probe("all_units_seeded_aa", topology, filtered, "Independent authored white/gray/dark glyph fill and genuine counter; actual two-unit canonical control plus exact-K graph", true);
  ASSERT_TRUE(topology.byteexact);ASSERT_EQ(topology.header_words.size(), 8u);EXPECT_EQ(topology.header_words[0u], 1u);EXPECT_EQ(topology.header_words[1u], 1u);
  for (UINT u = 2u; u < 8u; ++u) EXPECT_EQ(topology.header_words[u], 0u);
  const auto seeded = words(topology.artifacts.at("seeded.u32"));
  for (const UINT q : {60u * W_ + 79u, 62u * W_ + 79u, 60u * W_ + 90u}) {
    ASSERT_EQ(control[q], 1u);EXPECT_EQ(seeded[q] & 3u, 3u);EXPECT_EQ(filtered.filtered[q], 1u);EXPECT_EQ(filtered.rejected[q], 0u);
  }
  EXPECT_EQ(filtered.filtered, control);
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, PhysicalKTopologyResearchNonKWallCannotInventClosedCertificate) {
  if (!body_witness_research_configured()) GTEST_SKIP() << "Explicit private topology repository root required";
  const UINT q = author_topology_scene_counter(9921u, false);ASSERT_TRUE(run()) << error_;const auto control = words(outline_matte());
  const auto source = read(rgb_.resource.Get());auto copied = texture(W_, H_, DXGI_FORMAT_R16G16B16A16_FLOAT);context_->UpdateSubresource(copied.resource.Get(), 0u, nullptr, source.data(), W_ * 8u, 0u);
  // Kernel-only copied source opens the black ring with RGB58. Broad blacklike
  // still sees a wall, whereas exact physical K must not. This demonstrates the
  // graph mechanism, not a same-published-Source5 or positive alpha quality claim.
  const pixel16 dark {code_half(58u), code_half(58u), code_half(58u), 0x3c00u};
  const std::array<pixel16, 12u> corridor {dark, dark, dark, dark, dark, dark, dark, dark, dark, dark, dark, dark};
  const D3D11_BOX gap {73u, 106u, 0u, 75u, 112u, 1u};context_->UpdateSubresource(copied.resource.Get(), 0u, &gap, corridor.data(), 16u, 0u);
  const auto [topology, filtered] = probe_current_physical_k_topology(copied.srv.Get());
  export_inside_topology_probe("nonK_corridor", topology, filtered, "Kernel-only copied RGB58 opening: canonical broad graph remains closed, physical-K graph becomes exterior; producer source/mask unchanged", true);
  ASSERT_TRUE(topology.byteexact);ASSERT_EQ(control[q], 1u);
  const auto old_closed = words(topology.artifacts.at("canonical_control_closed_unseeded.u32"));const auto closed = words(topology.artifacts.at("closed_unseeded.u32"));const auto exterior = words(topology.artifacts.at("exterior.u32"));
  EXPECT_NE(old_closed[q] & 1u, 0u);EXPECT_EQ(closed[q] & 1u, 0u);EXPECT_NE(exterior[q] & 1u, 0u);
  EXPECT_EQ(filtered.rejected[q], 0u);EXPECT_EQ(filtered.filtered[q], 1u);EXPECT_EQ(read(rgb_.resource.Get()), source);EXPECT_EQ(words(outline_matte()), control);
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, PhysicalKTopologyResearchObservesSceneCounterAndRetainsSeededGlyph) {
  if (!body_witness_research_configured()) GTEST_SKIP() << "Explicit private topology repository root required";
  const UINT q = author_topology_scene_counter(9922u, false);ASSERT_TRUE(run()) << error_;const auto control = words(outline_matte());
  auto [topology, filtered] = probe_current_physical_k_topology();
  attach_physical_k_authored_alpha(topology, filtered, false);
  export_inside_topology_probe("closed_scene_counter", topology, filtered, "Independent authored alpha-zero scene ring below caption core; actual current BODY/M and exact-K graph supply mechanism prerequisites only", true);
  ASSERT_TRUE(topology.byteexact);ASSERT_EQ(words(body())[q], 1u);ASSERT_EQ(control[q], 1u);
  const auto closed = words(topology.artifacts.at("closed_unseeded.u32"));ASSERT_NE(closed[q] & 1u, 0u);
  EXPECT_NE(filtered.rejected[q] & 1u, 0u);EXPECT_EQ(filtered.filtered[q], 0u);EXPECT_EQ(filtered.filtered[60u * W_ + 90u], 1u);
  RecordProperty("physical_k_scene_counter_target_alpha", "0; independently authored scene, not semantic inference from graph");
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, PhysicalKTopologyResearchInvalidBoundaryRemainsUnknown) {
  if (!body_witness_research_configured()) GTEST_SKIP() << "Explicit private topology repository root required";
  const UINT q = author_topology_scene_counter(9923u, true);ASSERT_TRUE(run()) << error_;const auto control = words(outline_matte());const auto source = read(rgb_.resource.Get());
  auto copied = texture(W_, H_, DXGI_FORMAT_R16G16B16A16_FLOAT);context_->UpdateSubresource(copied.resource.Get(), 0u, nullptr, source.data(), W_ * 8u, 0u);
  const std::array<pixel16, 2u> invalid {{{0u, 0u, 0u, 0u}, {0u, 0u, 0u, 0u}}};const D3D11_BOX wall {73u, 108u, 0u, 75u, 109u, 1u};
  context_->UpdateSubresource(copied.resource.Get(), 0u, &wall, invalid.data(), 16u, 0u);const auto [topology, filtered] = probe_current_physical_k_topology(copied.srv.Get());
  export_inside_topology_probe("invalid_wall", topology, filtered, "Kernel-only copied invalid RGB wall cuts open scene corridor; invalid source cannot authorize physical-K closure", true);
  ASSERT_TRUE(topology.byteexact);const auto unknown = words(topology.artifacts.at("unknown.u32"));const auto closed = words(topology.artifacts.at("closed_unseeded.u32"));
  EXPECT_NE(unknown[q] & 1u, 0u);EXPECT_EQ(closed[q] & 1u, 0u);EXPECT_EQ(filtered.rejected[q], 0u);EXPECT_EQ(filtered.filtered[q], control[q]);EXPECT_EQ(read(rgb_.resource.Get()), source);
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, PhysicalKTopologyResearchCannotReuseSourceOrCanonicalControl) {
  if (!body_witness_research_configured()) GTEST_SKIP() << "Explicit private topology repository root required";
  author_band(5u, 9924u);ASSERT_TRUE(run()) << error_;const auto [topology_a, filtered_a] = probe_current_physical_k_topology();ASSERT_TRUE(topology_a.byteexact);
  author_band(5u, 9925u);ASSERT_TRUE(run()) << error_;const auto literal_b = words(outline_matte());const auto inputs_b = current_body_witness_inputs();
  const auto refused = probe_body_with_topology(cb_.resource.Get(), W_, H_, inputs_b, topology_a, true);
  export_inside_topology_probe("source_b_refuses_a", topology_a, refused, "Authenticated source B cannot consume cached physical topology A", true);
  EXPECT_FALSE(refused.metadata.at("filter_dispatched").get<bool>());EXPECT_FALSE(refused.metadata.at("topology_consumed_inputs_match").get<bool>());EXPECT_EQ(refused.filtered, literal_b);
  EXPECT_TRUE(std::all_of(refused.rejected.begin(), refused.rejected.end(), [](UINT v) { return v == 0u; }));EXPECT_EQ(words(refused.consumed.at("TopologyGate.u32"))[2u], 0u);
  // The original canonical control also cannot be rebound before physical emit.
  const auto e = helper_.enclosure_views();const auto r = helper_.raw_views();const auto canonical_b = replay_inside_topology(cb_.resource.Get(), W_, H_, inputs_b, e.all_body_bits.Get(), r.matte.Get(), e.matte.Get(), program_.inside_shaders, program_.roi_graph);
  ASSERT_TRUE(canonical_b.byteexact);const auto source_b = read(rgb_.resource.Get());auto copy = texture(W_, H_, DXGI_FORMAT_R16G16B16A16_FLOAT);context_->UpdateSubresource(copy.resource.Get(), 0u, nullptr, source_b.data(), W_ * 8u, 0u);
  const pixel16 changed {code_half(129u), code_half(128u), code_half(128u), 0x3c00u};const D3D11_BOX pixel {40u, 30u, 0u, 41u, 31u, 1u};context_->UpdateSubresource(copy.resource.Get(), 0u, &pixel, changed.data(), 8u, 0u);
  const auto changed_inputs = current_body_witness_inputs(copy.srv.Get());const auto mismatched = replay_physical_k_topology(cb_.resource.Get(), W_, H_, changed_inputs, canonical_b, program_.roi_graph);
  const auto rejected_control = probe_body_with_topology(cb_.resource.Get(), W_, H_, changed_inputs, mismatched, true);
  export_inside_topology_probe("copied_rgb_refuses_canonical_b", mismatched, rejected_control, "Kernel-only copied RGB changes one code; canonical replay from B does not authorize a new physical graph or filter", true);
  EXPECT_FALSE(mismatched.byteexact);EXPECT_FALSE(mismatched.metadata.at("canonical_control_consumed_inputs_match").get<bool>());EXPECT_FALSE(rejected_control.metadata.at("filter_dispatched").get<bool>());EXPECT_EQ(rejected_control.filtered, literal_b);
  EXPECT_TRUE(std::all_of(mismatched.header_words.begin(), mismatched.header_words.end(), [](UINT v) { return v == 0u; }));
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, PhysicalKTopologyResearchRetainsUnseededGlyphAmbiguityAsRejectedQuality) {
  if (!body_witness_research_configured()) GTEST_SKIP() << "Explicit private topology repository root required";
  // Same independently authored RGB geometry as the scene counter, now declared
  // a disconnected gray caption accent inside its original black stroke. There
  // are no exact qualified FG samples inside this component. Physical closure
  // cannot distinguish these two valid source-identical layer interpretations.
  for (const UINT fill_width : {8u, 2u}) {
    const UINT q = author_topology_scene_counter(9926u + fill_width, false);
    if (fill_width == 2u) {rect(70u, 101u, 78u, 107u, {0u, 0u, 0u, 255u});rect(73u, 101u, 75u, 107u, {160u, 160u, 160u, 255u});upload_pixels_from_origin();}
    ASSERT_TRUE(run()) << error_;const auto control = words(outline_matte());const auto body_words = words(body());auto [topology, filtered] = probe_current_physical_k_topology();
    attach_physical_k_authored_alpha(topology, filtered, true);
    const auto seeded = words(topology.artifacts.at("seeded.u32"));const auto closed = words(topology.artifacts.at("closed_unseeded.u32"));
    const bool prerequisites = topology.byteexact && control[q] == 1u && body_words[q] == 1u && (seeded[q] & 1u) == 0u && (closed[q] & 1u) != 0u;
    filtered.metadata["independent_authored_truth"]["gray_fill_width"] = fill_width;
    filtered.metadata["independent_authored_truth"]["actual_erasure_mechanism_prerequisites_met"] = prerequisites;
    filtered.metadata["independent_authored_truth"]["candidate_loses_visible_authored_ink"] = prerequisites && filtered.filtered[q] == 0u;
    filtered.metadata["quality_disposition"] = prerequisites && filtered.filtered[q] == 0u ? "REJECT_VISIBLE_AUTHORED_GLYPH_WITHDRAWAL" : "UNQUALIFIED_ADVERSARY_NO_ALPHA_CERTAINTY";
    const auto label = fill_width == 2u ? "thin_unseeded_accent_ambiguity" : "unseeded_glyph_ambiguity";
    export_inside_topology_probe(label, topology, filtered, "DECLARED source-identical ambiguity: gray detached glyph/accent alpha255 versus scene ring alpha0; no graph semantic certainty. Thin variant remains explicitly unqualified if producer prerequisites fail.", true);
    ASSERT_TRUE(topology.byteexact);
    if (fill_width == 8u) ASSERT_TRUE(prerequisites) << "Established full-size ambiguity must retain actual BODY/M and unseeded closure prerequisites";
    if (prerequisites) {EXPECT_NE(filtered.rejected[q] & 1u, 0u);EXPECT_EQ(filtered.filtered[q], 0u);}
  }
  RecordProperty("physical_k_unseeded_glyph_target_alpha", "255 independently declared ink; proposal loses genuine glyph under this interpretation");
  RecordProperty("physical_k_semantic_quality_disposition", "REJECT_GENERAL_ALPHA_CERTAINTY: source-identical scene/glyph ambiguity remains; mechanics pass is not production approval");
}
