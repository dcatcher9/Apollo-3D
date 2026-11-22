TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, WidthWitnessResearchErasesDeepSceneWithValidCurrentExteriorColour) {
  if (!body_witness_research_configured()) GTEST_SKIP() << "Explicit private width-witness repository root required";
  author_band(5u, 9960u);rect(38u, 48u, 62u, 82u, {0u, 0u, 0u, 255u});rect(42u, 52u, 58u, 78u, {160u, 160u, 160u, 255u});
  rect(8u, 65u, 12u, 66u, {160u, 160u, 160u, 255u});upload_pixels_from_origin();ASSERT_TRUE(run()) << error_;
  const UINT q = 65u * W_ + 50u, witness = 65u * W_ + 8u;const auto literal = words(outline_matte());const auto e = helper_.enclosure_views();
  auto [topology, filtered] = probe_width_guard_inputs(current_body_witness_inputs());attach_width_guard_scene_truth(topology, filtered, 50u, 65u);
  filtered.metadata["reason_bits"]["keep_no_current_row_colour_witness"] = 131072u;
  filtered.metadata["width_witness_fixture"] = {{"kind", "valid-current-same-row-outside-all-covers"}, {"target_xy", {50u, 65u}}, {"witness_xy", {8u, 65u}}, {"match_RGB8", {160u, 160u, 160u}}, {"semantic_alpha_certainty", false}};
  export_width_guard_probe("width_witness_valid_exterior", topology, filtered, "Active independent scene target beyond measured local K width; exact valid current same-row exterior RGB supplies additional necessary evidence only");
  ASSERT_TRUE(topology.byteexact);ASSERT_TRUE(filtered.metadata.at("filter_dispatched").get<bool>());ASSERT_EQ(calibration()[139u], 5u);ASSERT_EQ(calibration()[141u], 1u);
  ASSERT_EQ(words(body())[q], 1u);ASSERT_EQ(literal[q], 1u);ASSERT_EQ(words(read(e.cover_bits.Get()))[witness], 0u);
  ASSERT_NE(words(topology.artifacts.at("closed_unseeded.u32"))[q] & 1u, 0u);ASSERT_EQ(words(topology.artifacts.at("seeded.u32"))[q] & 1u, 0u);ASSERT_EQ(words(topology.artifacts.at("unknown.u32"))[q] & 1u, 0u);
  ASSERT_NE(filtered.reasons[q] & 8192u, 0u);ASSERT_EQ(filtered.reasons[q] & (16384u | 32768u | 65536u | 131072u), 0u);
  EXPECT_NE(filtered.rejected[q] & 1u, 0u);EXPECT_EQ(filtered.filtered[q], 0u);EXPECT_EQ(filtered.metadata.at("mask_delta").at("actual_M_additions"), 0u);
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, WidthWitnessResearchKeepsSameDeepSceneWithoutExteriorMatch) {
  if (!body_witness_research_configured()) GTEST_SKIP() << "Explicit private width-witness repository root required";
  author_band(5u, 9961u);rect(38u, 48u, 62u, 82u, {0u, 0u, 0u, 255u});rect(42u, 52u, 58u, 78u, {160u, 160u, 160u, 255u});upload_pixels_from_origin();ASSERT_TRUE(run()) << error_;
  const UINT q = 65u * W_ + 50u;const auto literal = words(outline_matte());auto [topology, filtered] = probe_width_guard_inputs(current_body_witness_inputs());
  attach_width_guard_scene_truth(topology, filtered, 50u, 65u);filtered.metadata["reason_bits"]["keep_no_current_row_colour_witness"] = 131072u;
  filtered.metadata["width_witness_fixture"] = {{"kind", "no-current-exterior-match"}, {"target_xy", {50u, 65u}}, {"exterior_authored_RGB8", {128u, 128u, 128u}}, {"semantic_alpha_certainty", false}};
  export_width_guard_probe("width_witness_no_exterior_match", topology, filtered, "Same active scene target/graph/local width, but exterior row has no exact current target colour. Lack of a positive witness must preserve literal M.");
  ASSERT_TRUE(topology.byteexact);ASSERT_TRUE(filtered.metadata.at("filter_dispatched").get<bool>());ASSERT_EQ(calibration()[139u], 5u);ASSERT_EQ(calibration()[141u], 1u);
  ASSERT_EQ(words(body())[q], 1u);ASSERT_EQ(literal[q], 1u);ASSERT_NE(words(topology.artifacts.at("closed_unseeded.u32"))[q] & 1u, 0u);
  ASSERT_EQ(words(topology.artifacts.at("seeded.u32"))[q] & 1u, 0u);ASSERT_EQ(words(topology.artifacts.at("unknown.u32"))[q] & 1u, 0u);
  ASSERT_NE(filtered.reasons[q] & 8192u, 0u);ASSERT_EQ(filtered.reasons[q] & (16384u | 32768u | 65536u), 0u);ASSERT_NE(filtered.reasons[q] & 131072u, 0u);
  EXPECT_EQ(filtered.rejected[q], 0u);EXPECT_EQ(filtered.filtered[q], 1u);EXPECT_EQ(filtered.filtered, literal);
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, WidthWitnessResearchRejectsOtherCoverOtherRowAndInvalidAlphaMatches) {
  if (!body_witness_research_configured()) GTEST_SKIP() << "Explicit private width-witness repository root required";
  UINT frame = 9962u;
  for (const std::string kind : {"another-cover", "another-row", "invalid-alpha"}) {
    SCOPED_TRACE(kind);author_band(5u, frame++);rect(38u, 48u, 62u, 82u, {0u, 0u, 0u, 255u});rect(42u, 52u, 58u, 78u, {160u, 160u, 160u, 255u});
    const UINT witness_y = kind == "another-row" ? 64u : 65u;rect(8u, witness_y, 12u, witness_y + 1u, {160u, 160u, 160u, 255u});
    if (kind == "another-cover") {
      ocr_words_[3u] = ocr_words_[4u] = 2u;uip_words_[3u] = 3u;
      const std::array<UINT, 4u> core {0u, 7u, 1u, 9u}, cover {0u, 6u, 1u, 10u};
      std::copy(core.begin(), core.end(), ocr_words_.begin() + 24u);std::copy(cover.begin(), cover.end(), ocr_words_.begin() + 152u);
    }
    upload_pixels_from_origin();ASSERT_TRUE(run()) << error_;const UINT q = 65u * W_ + 50u, witness = witness_y * W_ + 8u;
    const auto literal = words(outline_matte());const auto source = read(rgb_.resource.Get());const auto e = helper_.enclosure_views();const auto cover = words(read(e.cover_bits.Get()));auto inputs = current_body_witness_inputs();
    auto copied = texture(W_, H_, DXGI_FORMAT_R16G16B16A16_FLOAT);
    if (kind == "invalid-alpha") {
      // Explicit kernel-only invalid-source operand. Production Source5, mask
      // and receipts stay unchanged; all private replay/filter joins consume
      // this exact copied source, so a positive source-A certificate is not reused.
      context_->UpdateSubresource(copied.resource.Get(), 0u, nullptr, source.data(), W_ * 8u, 0u);
      const pixel16 invalid {code_half(160u), code_half(160u), code_half(160u), 0u};const std::array<pixel16, 4u> patch {invalid, invalid, invalid, invalid};
      const D3D11_BOX bounds {8u, 65u, 0u, 12u, 66u, 1u};context_->UpdateSubresource(copied.resource.Get(), 0u, &bounds, patch.data(), 32u, 0u);inputs[0u] = copied.srv.Get();
    }
    auto [topology, filtered] = probe_width_guard_inputs(inputs);attach_width_guard_scene_truth(topology, filtered, 50u, 65u);
    filtered.metadata["reason_bits"]["keep_no_current_row_colour_witness"] = 131072u;
    filtered.metadata["width_witness_fixture"] = {{"kind", kind}, {"target_xy", {50u, 65u}}, {"rejected_match_xy", {8u, witness_y}}, {"match_RGB8", {160u, 160u, 160u}}, {"kernel_only_alpha_override", kind == "invalid-alpha"}, {"semantic_alpha_certainty", false}};
    const auto label = std::string("width_witness_") + kind;export_width_guard_probe(label.c_str(), topology, filtered, "Actual active closed/strong-local-width scene target; a match under another OCR cover, off the native row, or with invalid source alpha cannot authorize erasure");
    ASSERT_TRUE(topology.byteexact);ASSERT_TRUE(filtered.metadata.at("filter_dispatched").get<bool>());ASSERT_EQ(calibration()[139u], 5u);ASSERT_EQ(calibration()[141u], 1u);
    ASSERT_EQ(words(body())[q], 1u);ASSERT_EQ(literal[q], 1u);ASSERT_NE(words(topology.artifacts.at("closed_unseeded.u32"))[q] & 1u, 0u);
    ASSERT_EQ(words(topology.artifacts.at("seeded.u32"))[q] & 1u, 0u);ASSERT_EQ(words(topology.artifacts.at("unknown.u32"))[q] & 1u, 0u);
    if (kind == "another-cover") {ASSERT_NE(cover[witness] & 2u, 0u);ASSERT_EQ(cover[witness] & 1u, 0u);ASSERT_EQ(cover[q], 1u);}
    else ASSERT_EQ(cover[witness], 0u);
    ASSERT_NE(filtered.reasons[q] & 8192u, 0u);ASSERT_EQ(filtered.reasons[q] & (16384u | 32768u | 65536u), 0u);ASSERT_NE(filtered.reasons[q] & 131072u, 0u);
    EXPECT_EQ(filtered.rejected[q], 0u);EXPECT_EQ(filtered.filtered[q], 1u);
    // A valid patch on row64 may authorize other scene pixels on row64; it
    // cannot authorize this row65 target. The two other negatives have no
    // valid exterior match on any row containing this authored scene colour.
    if (kind != "another-row") EXPECT_EQ(filtered.filtered, literal);
    EXPECT_EQ(read(rgb_.resource.Get()), source);EXPECT_EQ(words(outline_matte()), literal);
  }
}
