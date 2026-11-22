TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, OverlapWidthResearchBorrowsCompatibleBoxesAcrossDifferentObservedHeights) {
  if (!overlap_width_configured() || !shared_width_research_configured()) GTEST_SKIP() << "Explicit private overlap/strict repository roots required";
  const auto truth = author_overlap_width_case(false, 10010u);ASSERT_TRUE(run()) << error_;const auto original_m = words(outline_matte()), original_f2 = calibration();const auto e = helper_.enclosure_views();
  auto boxes = words(read(e.family_models.Get()));const UINT offset = 17u * truth.receiver;ASSERT_EQ((boxes[offset] >> 8u) & 255u, 1u);boxes[offset + 1u] = 0x00fefefeu;
  auto families = texture(17u, 8u, DXGI_FORMAT_R32_UINT);context_->UpdateSubresource(families.resource.Get(), 0u, nullptr, boxes.data(), 17u * 4u, 0u);const auto inputs = current_overlap_width_inputs(nullptr, families.srv.Get());
  auto learned = learn_overlap_width(cb_.resource.Get(), W_, H_, inputs);const auto candidate = consume_overlap_width(cb_.resource.Get(), W_, H_, inputs, learned);
  std::array<ID3D11ShaderResourceView *, 8u> strict_inputs {};std::copy_n(inputs.begin(), 8u, strict_inputs.begin());const auto strict = probe_shared_width(cb_.resource.Get(), W_, H_, strict_inputs);
  learned.metadata["strict_prior_control"] = {{"same_source_SRV_bytes", true}, {"source_closure", strict.metadata.at("source_closure")}, {"prior_words_row_major", strict.prior}, {"candidate_bytes_sha256", hash_bytes(raw(strict.candidate))}};
  export_overlap_width_probe("overlap_width_matching_height_variation", learned, candidate, &truth, "Actual current donor/UNKNOWN recipient source; different observed core heights; kernel-only receiver FG box widened from white255 to254..255 to test nonidentical overlap. No producer table mutation.");
  ASSERT_EQ(learned.prior_words[8u], 1u);ASSERT_EQ(learned.prior_words[0u], 5u);ASSERT_NE(original_f2[128u + 64u * truth.receiver + 13u], 1u);
  ASSERT_NE(learned.prior_words[32u], learned.prior_words[32u + truth.receiver]);ASSERT_EQ(learned.prior_words[8u + truth.receiver], 2u);ASSERT_EQ(learned.prior_words[truth.receiver], 5u);
  ASSERT_TRUE(candidate.metadata.at("candidate_dispatched").get<bool>());const UINT bit = 1u << truth.receiver;const auto found = std::find_if(candidate.candidate.begin(), candidate.candidate.end(), [&](UINT v) { return (v & bit) != 0u; });ASSERT_NE(found, candidate.candidate.end());
  const UINT q = UINT(found - candidate.candidate.begin());ASSERT_EQ(original_m[q], 0u);ASSERT_GT(truth.alpha[q], 0u);EXPECT_EQ(candidate.union_m[q], 1u);EXPECT_EQ(origin_pixels_[q][0], 0u);EXPECT_EQ(truth.scene[q][0], 1u);
  EXPECT_NE(strict.prior[8u + truth.receiver], 2u);EXPECT_TRUE(std::all_of(strict.candidate.begin(), strict.candidate.end(), [](UINT v) { return v == 0u; }));
  for (std::size_t i = 0u; i < candidate.union_m.size(); ++i) {if (original_m[i]) EXPECT_EQ(candidate.union_m[i], 1u);if (candidate.candidate[i]) EXPECT_GT(truth.alpha[i], 0u);}
  EXPECT_EQ(calibration(), original_f2);EXPECT_EQ(words(outline_matte()), original_m);
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, OverlapWidthResearchAbstainsOnConflictingStrongDonorWidths) {
  if (!overlap_width_configured()) GTEST_SKIP() << "Explicit private overlap repository root required";
  const auto truth = author_overlap_width_case(true, 10011u);ASSERT_TRUE(run()) << error_;const auto literal = words(outline_matte());const auto inputs = current_overlap_width_inputs();
  const auto learned = learn_overlap_width(cb_.resource.Get(), W_, H_, inputs);const auto candidate = consume_overlap_width(cb_.resource.Get(), W_, H_, inputs, learned);
  export_overlap_width_probe("overlap_width_conflicting_donors", learned, candidate, &truth, "Actual strong donor widths5/7, same current foreground/K style, UNKNOWN dark recipient; disagreement cannot supply a borrowed width");
  ASSERT_EQ(learned.prior_words[8u], 1u);ASSERT_EQ(learned.prior_words[9u], 1u);ASSERT_NE(learned.prior_words[0u], learned.prior_words[1u]);ASSERT_NE(calibration()[128u + 64u * truth.receiver + 13u], 1u);
  EXPECT_EQ(learned.prior_words[8u + truth.receiver], 3u);EXPECT_EQ(learned.prior_words[truth.receiver], 0u);EXPECT_EQ(candidate.union_m, literal);EXPECT_TRUE(std::all_of(candidate.candidate.begin(), candidate.candidate.end(), [](UINT v) { return v == 0u; }));
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, OverlapWidthResearchRejectsAmbiguousAndDisjointForegroundCorrespondence) {
  if (!overlap_width_configured()) GTEST_SKIP() << "Explicit private overlap repository root required";
  const auto truth = author_overlap_width_case(false, 10012u);ASSERT_TRUE(run()) << error_;const auto literal = words(outline_matte());const auto original = words(read(helper_.enclosure_views().family_models.Get()));
  for (const bool ambiguous : {true, false}) {
    auto models = original;
    if (ambiguous) for (UINT unit = 0u; unit < 2u; ++unit) {
      std::fill_n(models.begin() + 17u * unit, 17u, 0u);models[17u * unit] = 3u | (3u << 8u);
      models[17u * unit + 1u] = 0x00f0f0f0u;models[17u * unit + 2u] = 0x00fafafau;
      models[17u * unit + 3u] = 0x00f5f5f5u;models[17u * unit + 4u] = 0x00ffffffu;
    } else {models[17u * truth.receiver + 1u] = 0x0000fafau;models[17u * truth.receiver + 2u] = 0x0002ffffu;}
    auto family = texture(17u, 8u, DXGI_FORMAT_R32_UINT);context_->UpdateSubresource(family.resource.Get(), 0u, nullptr, models.data(), 17u * 4u, 0u);const auto inputs = current_overlap_width_inputs(nullptr, family.srv.Get());
    const auto learned = learn_overlap_width(cb_.resource.Get(), W_, H_, inputs);const auto candidate = consume_overlap_width(cb_.resource.Get(), W_, H_, inputs, learned);
    export_overlap_width_probe(ambiguous ? "overlap_width_ambiguous_FG" : "overlap_width_disjoint_FG", learned, candidate, nullptr, "Kernel-only copied palette adversary; exact current source/proofs remain read-only. Nonunique overlap or disjoint qualified families cannot authorize borrowing.");
    ASSERT_EQ(learned.prior_words[8u], 1u);EXPECT_EQ(learned.prior_words[8u + truth.receiver], ambiguous ? 8u : 4u);EXPECT_NE(learned.prior_words[8u * 8u + truth.receiver] & (ambiguous ? 64u : 8u), 0u);
    EXPECT_EQ(candidate.union_m, literal);EXPECT_TRUE(std::all_of(candidate.candidate.begin(), candidate.candidate.end(), [](UINT v) { return v == 0u; }));
  }
  EXPECT_EQ(words(read(helper_.enclosure_views().family_models.Get())), original);
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, OverlapWidthResearchCurrentExteriorColourVetoRevokesAddition) {
  if (!overlap_width_configured()) GTEST_SKIP() << "Explicit private overlap repository root required";
  const auto truth = author_overlap_width_case(false, 10013u);ASSERT_TRUE(run()) << error_;const auto inputs = current_overlap_width_inputs();const auto learned = learn_overlap_width(cb_.resource.Get(), W_, H_, inputs);const auto control = consume_overlap_width(cb_.resource.Get(), W_, H_, inputs, learned);
  ASSERT_EQ(learned.prior_words[8u + truth.receiver], 2u);const UINT bit = 1u << truth.receiver;const auto first = std::find_if(control.candidate.begin(), control.candidate.end(), [&](UINT v) { return (v & bit) != 0u; });ASSERT_NE(first, control.candidate.end());const UINT q = UINT(first - control.candidate.begin()), y = q / W_;
  ASSERT_EQ(words(outline_matte())[q], 0u);ASSERT_EQ(words(read(helper_.enclosure_views().cover_bits.Get()))[y * W_], 0u);const auto source = read(rgb_.resource.Get());auto rgb = texture(W_, H_, DXGI_FORMAT_R16G16B16A16_FLOAT);context_->UpdateSubresource(rgb.resource.Get(), 0u, nullptr, source.data(), W_ * 8u, 0u);
  const pixel16 black {0u, 0u, 0u, 0x3c00u};const D3D11_BOX pixel {0u, y, 0u, 1u, y + 1u, 1u};context_->UpdateSubresource(rgb.resource.Get(), 0u, &pixel, black.data(), 8u, 0u);const auto changed_inputs = current_overlap_width_inputs(rgb.srv.Get());
  const auto relearned = learn_overlap_width(cb_.resource.Get(), W_, H_, changed_inputs);const auto vetoed = consume_overlap_width(cb_.resource.Get(), W_, H_, changed_inputs, relearned);
  export_overlap_width_probe("overlap_width_current_exterior_veto", relearned, vetoed, nullptr, "Kernel-only copied current RGB adds exact black outside all covers on an otherwise active candidate row. Fresh learning/consumption of this same changed tuple must veto addition without mutating producer RGB/M.");
  ASSERT_TRUE(vetoed.metadata.at("candidate_dispatched").get<bool>());ASSERT_EQ(relearned.prior_words[8u + truth.receiver], 2u);EXPECT_EQ(vetoed.candidate[q] & bit, 0u);EXPECT_EQ(vetoed.union_m[q], 0u);EXPECT_EQ(read(rgb_.resource.Get()), source);
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, OverlapWidthResearchRejectsCachedSourceAndChangedOriginalMask) {
  if (!overlap_width_configured()) GTEST_SKIP() << "Explicit private overlap repository root required";
  author_overlap_width_case(false, 10014u);ASSERT_TRUE(run()) << error_;const auto a = learn_overlap_width(cb_.resource.Get(), W_, H_, current_overlap_width_inputs());
  author_overlap_width_case(false, 10015u);ASSERT_TRUE(run()) << error_;const auto literal = words(outline_matte());const auto inputs = current_overlap_width_inputs();const auto stale = consume_overlap_width(cb_.resource.Get(), W_, H_, inputs, a);
  export_overlap_width_probe("overlap_width_source_B_refuses_A", a, stale, nullptr, "Actual current source B cannot consume source A's cached valid prior. Candidate stays GPU-zero; union is a GPU literal copy of B's original M.");
  EXPECT_FALSE(stale.metadata.at("candidate_dispatched").get<bool>());EXPECT_FALSE(stale.metadata.at("source_tuple_matches_learning").get<bool>());EXPECT_EQ(stale.union_m, literal);EXPECT_TRUE(std::all_of(stale.candidate.begin(), stale.candidate.end(), [](UINT v) { return v == 0u; }));
  auto b = learn_overlap_width(cb_.resource.Get(), W_, H_, inputs);const auto source = read(rgb_.resource.Get());auto rgb = texture(W_, H_, DXGI_FORMAT_R16G16B16A16_FLOAT);context_->UpdateSubresource(rgb.resource.Get(), 0u, nullptr, source.data(), W_ * 8u, 0u);
  const pixel16 change {code_half(129u), code_half(128u), code_half(128u), 0x3c00u};const D3D11_BOX point {0u, 0u, 0u, 1u, 1u, 1u};context_->UpdateSubresource(rgb.resource.Get(), 0u, &point, change.data(), 8u, 0u);
  const auto changed = consume_overlap_width(cb_.resource.Get(), W_, H_, current_overlap_width_inputs(rgb.srv.Get()), b);export_overlap_width_probe("overlap_width_changed_RGB_refuses_B", b, changed, nullptr, "Same C24/SCT with a different copied current RGB is not the tuple that learned the prior");
  EXPECT_FALSE(changed.metadata.at("candidate_dispatched").get<bool>());EXPECT_EQ(changed.union_m, literal);EXPECT_TRUE(std::all_of(changed.candidate.begin(), changed.candidate.end(), [](UINT v) { return v == 0u; }));
  ASSERT_EQ(literal[0u], 0u);auto mask = texture(W_, H_, DXGI_FORMAT_R32_UINT);context_->UpdateSubresource(mask.resource.Get(), 0u, nullptr, literal.data(), W_ * 4u, 0u);const UINT one = 1u;context_->UpdateSubresource(mask.resource.Get(), 0u, &point, &one, 4u, 0u);
  const auto changed_m = consume_overlap_width(cb_.resource.Get(), W_, H_, current_overlap_width_inputs(nullptr, nullptr, mask.srv.Get()), b);export_overlap_width_probe("overlap_width_changed_M_refuses_B", b, changed_m, nullptr, "Kernel-only copied original M changes one word; fallback must use that current GPU mask, never source A/B's remembered mask");
  EXPECT_FALSE(changed_m.metadata.at("candidate_dispatched").get<bool>());EXPECT_EQ(changed_m.union_m, words(read(mask.resource.Get())));EXPECT_TRUE(std::all_of(changed_m.candidate.begin(), changed_m.candidate.end(), [](UINT v) { return v == 0u; }));
  ASSERT_NE(b.prior_words[1u], 0u);const UINT zero = 0u;const D3D11_BOX table_word {1u, 0u, 0u, 2u, 1u, 1u};context_->UpdateSubresource(b.prior.resource.Get(), 0u, &table_word, &zero, 4u, 0u);
  const auto changed_table = consume_overlap_width(cb_.resource.Get(), W_, H_, inputs, b);export_overlap_width_probe("overlap_width_changed_table_refuses_B", b, changed_table, nullptr, "Private prior texture bytes changed after learning; even identical source inputs must retain literal M");
  EXPECT_FALSE(changed_table.metadata.at("candidate_dispatched").get<bool>());EXPECT_FALSE(changed_table.metadata.at("prior_bytes_match_learning").get<bool>());EXPECT_EQ(changed_table.union_m, literal);
  EXPECT_TRUE(std::all_of(changed_table.candidate.begin(), changed_table.candidate.end(), [](UINT v) { return v == 0u; }));EXPECT_EQ(read(rgb_.resource.Get()), source);EXPECT_EQ(words(outline_matte()), literal);
}
