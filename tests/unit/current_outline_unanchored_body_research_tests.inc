TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, UnanchoredBodyResearchPreservesCurrentInnerAaAndBlackSupport) {
  if (!body_witness_research_configured()) GTEST_SKIP() << "Explicit private body-witness repository root required";
  author_band(5u, 9800u);
  // Independently authored inner-edge colours, not outputs copied from a mask.
  // The midgray cell is nonblacklike; the dark cell retains existing support.
  rect(79u, 60u, 80u, 61u, {128u, 128u, 128u, 255u});
  rect(79u, 62u, 80u, 63u, {64u, 64u, 64u, 255u});upload_pixels_from_origin();
  ASSERT_TRUE(run()) << error_;const auto original = words(outline_matte());const auto before_body = body();
  const auto probe = probe_body_witness_current();
  export_body_witness_probe("inner_aa_black", probe, "Actual unchanged producer; authored current midgray/dark inner-edge samples, no alpha reconstruction");
  ASSERT_EQ(original[60u * W_ + 79u], 1u);ASSERT_EQ(original[62u * W_ + 79u], 1u);
  EXPECT_EQ(probe.filtered[60u * W_ + 79u], 1u);EXPECT_EQ(probe.filtered[62u * W_ + 79u], 1u);
  EXPECT_EQ(probe.filtered, original);EXPECT_EQ(body(), before_body);
  EXPECT_GT(std::count_if(probe.reasons.begin(), probe.reasons.end(), [](UINT word) { return (word & 4u) != 0u; }), 0);
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, UnanchoredBodyResearchKeepsPointwiseForegroundOnOpenPhysicalInterior) {
  if (!body_witness_research_configured()) GTEST_SKIP() << "Explicit private body-witness repository root required";
  author_band(5u, 9801u);
  // Current colour ownership remains white-only. The nonwhite opening joins
  // the first glyph's physical nonblack interior to the actual cover edge;
  // the second closed glyph retains the unit's actual adopted model.
  rect(95u, 86u, 96u, 113u, {255u, 160u, 255u, 255u});upload_pixels_from_origin();
  ASSERT_TRUE(run()) << error_;const auto original = words(outline_matte());
  const auto inside = words(read(helper_.mutable_views().inside_bits.Get()));
  const auto current_body = words(body());const auto probe = probe_body_witness_current();
  export_body_witness_probe("open_inner", probe, "Actual current nonwhite opening; exact qualified current foreground must not require a closed Inside component");
  constexpr UINT x = 90u, y = 60u;const UINT q = y * W_ + x;
  ASSERT_EQ(current_body[q], 1u);ASSERT_EQ(inside[q] & 1u, 0u);ASSERT_EQ(original[q], 1u);
  EXPECT_EQ(probe.rejected[q], 0u);EXPECT_EQ(probe.filtered[q], 1u);EXPECT_NE(probe.reasons[q] & 4u, 0u);
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, UnanchoredBodyResearchRetainsCounterAndMultipleUnitKeepEvidence) {
  if (!body_witness_research_configured()) GTEST_SKIP() << "Explicit private body-witness repository root required";
  const auto truth = author_shared_width_case("counter_scene", false, 9802u);ASSERT_TRUE(run()) << error_;
  const auto original = words(outline_matte());const auto probe = probe_body_witness_current();
  export_body_witness_probe("two_unit_counter", probe, "Actual two-unit source/Inside graph; source-identical counter is not newly classified as scene");
  EXPECT_EQ(probe.filtered, original);
  for (const UINT q : {(60u * W_ + 55u), (60u * W_ + truth.receiver_left + 55u)}) {
    ASSERT_EQ(original[q], 1u);EXPECT_EQ(probe.filtered[q], 1u);EXPECT_EQ(probe.rejected[q], 0u);
  }
  // Separately exercise another covered unit's valid Inside/foreground keep.
  // This input-copy perturbation does not mutate or relabel producer outputs.
  author_band(5u, 9803u);ocr_words_[3u] = ocr_words_[4u] = 2u;uip_words_[3u] = 3u;
  std::copy_n(ocr_words_.begin() + 16u, 4u, ocr_words_.begin() + 24u);
  std::copy_n(ocr_words_.begin() + 144u, 4u, ocr_words_.begin() + 152u);upload();ASSERT_TRUE(run()) << error_;
  auto selected = words(read(helper_.enclosure_views().selected_bits.Get()));
  for (auto &word : selected) word &= ~2u;
  auto copy = texture(W_, H_, DXGI_FORMAT_R32_UINT);context_->UpdateSubresource(copy.resource.Get(), 0u, nullptr, selected.data(), W_ * 4u, 0u);
  const auto other = probe_body_witness_current(nullptr, copy.srv.Get());
  export_body_witness_probe("other_unit_input_copy", other, "Kernel-only copied LOW input clears unit1 ownership; actual current unit1 Inside/qualified foreground still supplies conservative keep; producer remains unchanged");
  const UINT q = 60u * W_ + 90u;
  EXPECT_EQ(other.rejected[q], 0u);EXPECT_EQ(other.filtered[q], 1u);EXPECT_NE(other.reasons[q] & 32u, 0u);
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, UnanchoredBodyResearchStaleOrUnavailableInsideKeepsLiteralCurrentMask) {
  if (!body_witness_research_configured()) GTEST_SKIP() << "Explicit private body-witness repository root required";
  author_band(5u, 9804u);ASSERT_TRUE(run()) << error_;const auto original = words(outline_matte());
  sct_words_[4u] ^= 1u;context_->UpdateSubresource(sct_.resource.Get(), 0u, nullptr, sct_words_.data(), 0u, 0u);
  const auto stale = probe_body_witness_current();
  EXPECT_EQ(stale.filtered, original);EXPECT_TRUE(std::all_of(stale.rejected.begin(), stale.rejected.end(), [](UINT x) { return x == 0u; }));
  sct_words_[4u] ^= 1u;context_->UpdateSubresource(sct_.resource.Get(), 0u, nullptr, sct_words_.data(), 0u, 0u);
  auto unavailable = texture(1u, 1u, DXGI_FORMAT_R32_UINT);
  const auto missing = probe_body_witness_current(unavailable.srv.Get());
  EXPECT_EQ(missing.filtered, original);EXPECT_TRUE(std::all_of(missing.rejected.begin(), missing.rejected.end(), [](UINT x) { return x == 0u; }));
  export_body_witness_probe("unavailable_inside", missing, "Explicit wrong-shape optional Inside input; literal current M retained and no rejection permission");
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, UnanchoredBodyResearchIndependentOtherInsideSurvivesUnavailablePhysicalAdoption) {
  if (!body_witness_research_configured()) GTEST_SKIP() << "Explicit private body-witness repository root required";
  author_band(5u, 9805u);rect(79u, 60u, 80u, 61u, {128u, 128u, 128u, 255u});
  ocr_words_[3u] = ocr_words_[4u] = 2u;uip_words_[3u] = 3u;
  std::copy_n(ocr_words_.begin() + 16u, 4u, ocr_words_.begin() + 24u);
  std::copy_n(ocr_words_.begin() + 144u, 4u, ocr_words_.begin() + 152u);upload_pixels_from_origin();ASSERT_TRUE(run()) << error_;
  const auto e = helper_.enclosure_views();const auto f = helper_.mutable_views();const UINT q = 60u * W_ + 79u;
  auto selected = words(read(e.selected_bits.Get())), inside = words(read(f.inside_bits.Get()));
  auto cbc = words(read(e.receipt.Get())), families = words(read(e.family_models.Get()));
  ASSERT_NE(selected[q] & 1u, 0u);ASSERT_NE(inside[q] & 2u, 0u);ASSERT_EQ(words(body())[q], 1u);
  for (auto &word : selected) word &= ~2u;for (auto &word : inside) word &= ~1u;
  // Explicit copied-table adversary: Inside's canonical readiness does not
  // depend on CBC selectionStatus. Remove only stricter physical adoption for
  // nonowner unit1 while retaining its real current fault-free Inside output.
  cbc[128u + 32u + 6u] = 2u;
  UINT white_mode = 8u;
  for (UINT mode = 0u; mode < 8u; ++mode) if ((families[0u] & (1u << (8u + mode))) != 0u &&
      families[2u + 2u * mode] == 0x00ffffffu) {white_mode = mode;break;}
  ASSERT_LT(white_mode, 8u);families[0u] = (families[0u] & 255u) | (1u << (8u + white_mode));
  auto selected_copy = texture(W_, H_, DXGI_FORMAT_R32_UINT), inside_copy = texture(W_, H_, DXGI_FORMAT_R32_UINT), family_copy = texture(17u, 8u, DXGI_FORMAT_R32_UINT);
  auto cbc_copy = buffer(2048u);
  context_->UpdateSubresource(selected_copy.resource.Get(), 0u, nullptr, selected.data(), W_ * 4u, 0u);
  context_->UpdateSubresource(inside_copy.resource.Get(), 0u, nullptr, inside.data(), W_ * 4u, 0u);
  context_->UpdateSubresource(family_copy.resource.Get(), 0u, nullptr, families.data(), 17u * 4u, 0u);
  context_->UpdateSubresource(cbc_copy.resource.Get(), 0u, nullptr, cbc.data(), 0u, 0u);
  const auto probe = probe_body_witness(cb_.resource.Get(), W_, H_, {rgb_.srv.Get(), ocr_.srv.Get(), selected_copy.srv.Get(),
    e.cover_bits.Get(), cbc_copy.srv.Get(), f.calibration_data.Get(), family_copy.srv.Get(), sct_.srv.Get(),
    inside_copy.srv.Get(), e.body.Get(), f.matte.Get()});
  export_body_witness_probe("independent_other_inside", probe, "Kernel-only COPIED selected/Inside/CBC/family adversary; independent nonowner current Inside survives unavailable physical adoption; producer inputs remain unchanged");
  EXPECT_EQ(probe.rejected[q], 0u);EXPECT_EQ(probe.filtered[q], 1u);
  EXPECT_NE(probe.reasons[q] & 8u, 0u);EXPECT_NE(probe.reasons[q] & 32u, 0u);EXPECT_NE(probe.reasons[q] & 256u, 0u);
  EXPECT_EQ(probe.consumed.at("InsideBits.u32"), raw(inside));EXPECT_EQ(probe.consumed.at("SelectedBits.u32"), raw(selected));
  EXPECT_EQ(probe.consumed.at("CBC.u32"), raw(cbc));EXPECT_EQ(probe.consumed.at("FamilyModels.u32"), raw(families));
  EXPECT_NE(probe.consumed.at("InsideBits.u32"), read(f.inside_bits.Get()));
}
