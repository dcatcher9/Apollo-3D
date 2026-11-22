// The source-only hard regions intentionally include source-identical scene.
// Green mechanics tests do not imply that the diagnostic candidate is alpha.
TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, SharedWidthPriorResearchSharesMatchingCurrentStyleWithoutMutation) {
  if (!shared_width_research_configured()) GTEST_SKIP() << "Explicit private shared-width repository root required";
  std::vector<UINT> forward;
  for (const bool reverse : {false, true}) {
    const auto truth = author_shared_width_case("matching", reverse, reverse ? 9701u : 9700u);
    ASSERT_TRUE(run()) << error_;expect_ready();expect_ring_ready();
    const auto before_data = calibration();const auto before_m = outline_matte();
    const auto probe = probe_shared_width_current();export_shared_width_case(truth, probe);
    ASSERT_EQ(probe.prior.size(), 128u);
    const UINT donor = reverse ? 1u : 0u, receiver = truth.receiver;
    ASSERT_EQ(probe.prior[8u + donor], 1u) << "Actual source must supply a strong easy donor";
    ASSERT_EQ(probe.prior[donor], 5u);
    ASSERT_GE(probe.prior[40u + donor], 2u);
    ASSERT_NE(probe.prior[24u + receiver], 1u) << "Hard receiver must not teach its own KNOWN width";
    ASSERT_EQ(probe.prior[8u + receiver], 2u) << "Matching current style must receive a separate prior";
    EXPECT_EQ(probe.prior[receiver], 5u);EXPECT_EQ(probe.prior[16u + receiver], 1u << donor);
    EXPECT_EQ(calibration(), before_data);EXPECT_EQ(outline_matte(), before_m);
    std::vector<UINT> collapsed(probe.candidate.size());
    std::transform(probe.candidate.begin(), probe.candidate.end(), collapsed.begin(), [](UINT word) { return UINT(word != 0u); });
    ASSERT_GT(std::count(collapsed.begin(), collapsed.end(), 1u), 0);
    if (!reverse) forward = collapsed;else EXPECT_EQ(collapsed, forward) << "Unit order must not change source candidate geometry";
  }
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, SharedWidthPriorResearchRejectsConflictingWidthsAndDifferentStyle) {
  if (!shared_width_research_configured()) GTEST_SKIP() << "Explicit private shared-width repository root required";
  UINT frame = 9710u;
  for (const std::string kind : {"mixed_widths", "mixed_style", "missing_fill"}) {
    const auto truth = author_shared_width_case(kind, false, frame++);ASSERT_TRUE(run()) << error_;
    const auto probe = probe_shared_width_current();export_shared_width_case(truth, probe);
    const UINT receiver = truth.receiver;
    ASSERT_EQ(probe.prior[8u], 1u) << "Negative must retain an active strong donor";
    if (kind == "mixed_widths") {
      ASSERT_EQ(probe.prior[9u], 1u);ASSERT_NE(probe.prior[0u], probe.prior[1u]);
      EXPECT_EQ(probe.prior[8u + receiver], 3u);EXPECT_EQ(probe.prior[receiver], 0u);
      EXPECT_EQ(probe.prior[16u + receiver], 3u);
    } else EXPECT_NE(probe.prior[8u + receiver], 2u);
    for (const UINT word : probe.candidate) EXPECT_EQ(word & (1u << receiver), 0u);
  }
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, SharedWidthPriorResearchRecordsAttachedSceneAndCounterAmbiguity) {
  if (!shared_width_research_configured()) GTEST_SKIP() << "Explicit private shared-width repository root required";
  UINT frame = 9720u;
  for (const std::string kind : {"matching", "counter_scene"}) {
    const auto truth = author_shared_width_case(kind, false, frame++);ASSERT_TRUE(run()) << error_;
    const auto probe = probe_shared_width_current();const auto report = export_shared_width_case(truth, probe);
    ASSERT_EQ(probe.prior[8u + truth.receiver], 2u);
    EXPECT_GT(report.at("candidate_truth_counts").at("ink").get<UINT>(), 0u);
    const auto &counts = report.at("candidate_truth_counts");
    EXPECT_EQ(counts.at("new_scene_vs_production").get<UINT>(), 0u);
    EXPECT_EQ(counts.at("ink_visible_alpha_ge5_source_delta_ge3").get<UINT>(), 0u);
    EXPECT_EQ(counts.at("new_ink_visible_alpha_ge5_source_delta_ge3").get<UINT>(), 0u);
    EXPECT_EQ(counts.at("ink_source_identical"), counts.at("ink"));
    EXPECT_EQ(counts.at("new_ink_source_identical"), counts.at("new_ink_vs_production"));
    // The solid authored glyph's measured band contains no scene; the
    // transparent counter supplies a separate semantic counterexample. Its
    // false candidate was already protected by production M, not a new FP.
    if (kind == "counter_scene") {
      EXPECT_GT(counts.at("scene").get<UINT>(), 0u);
      EXPECT_EQ(report.at("quality_status"), "REJECTED_AS_ALPHA_MASK_ATTACHED_SCENE_AMBIGUITY");
    } else {
      EXPECT_EQ(counts.at("scene").get<UINT>(), 0u);
      EXPECT_EQ(report.at("quality_status"), "MECHANICS_ONLY_NO_GENERAL_QUALITY_ACCEPTANCE");
    }
    EXPECT_FALSE(report.at("quality_accepted").get<bool>());
  }
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, SharedWidthPriorResearchRejectsStaleSourceProofAndRevokesChangedK) {
  if (!shared_width_research_configured()) GTEST_SKIP() << "Explicit private shared-width repository root required";
  const auto truth = author_shared_width_case("source_guards", false, 9730u);ASSERT_TRUE(run()) << error_;
  const auto good = probe_shared_width_current();export_shared_width_case(truth, good);
  ASSERT_EQ(good.prior[8u + truth.receiver], 2u);
  // Every stale invocation actively overwrites its own prior/candidate to zero.
  sct_words_[4u] ^= 1u;context_->UpdateSubresource(sct_.resource.Get(), 0u, nullptr, sct_words_.data(), 0u, 0u);
  const auto stale = probe_shared_width_current();
  EXPECT_TRUE(std::all_of(stale.prior.begin(), stale.prior.end(), [](UINT x) { return x == 0u; }));
  EXPECT_TRUE(std::all_of(stale.candidate.begin(), stale.candidate.end(), [](UINT x) { return x == 0u; }));
  sct_words_[4u] ^= 1u;context_->UpdateSubresource(sct_.resource.Get(), 0u, nullptr, sct_words_.data(), 0u, 0u);
  const auto found = std::find_if(good.candidate.begin(), good.candidate.end(), [&](UINT x) { return (x & (1u << truth.receiver)) != 0u; });
  ASSERT_NE(found, good.candidate.end());const UINT index = UINT(found - good.candidate.begin());
  const pixel16 changed {code_half(128u), code_half(128u), code_half(128u), 0x3c00u};
  const D3D11_BOX pixel {index % W_, index / W_, 0u, index % W_ + 1u, index / W_ + 1u, 1u};
  context_->UpdateSubresource(rgb_.resource.Get(), 0u, &pixel, changed.data(), 8u, 0u);
  const auto revoked = probe_shared_width_current();
  EXPECT_EQ(revoked.candidate[index] & (1u << truth.receiver), 0u);
  RecordProperty("shared_width_source_guard_scope", "Current SCT identity mismatch clears private outputs; current source non-K revokes the original-K candidate. This is not a new source snapshot or a content hash proof.");
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, SharedWidthPriorResearchIgnoresPaletteOrderAndRejectsDuplicateQualifiedBoxes) {
  if (!shared_width_research_configured()) GTEST_SKIP() << "Explicit private shared-width repository root required";
  const auto truth = author_shared_width_case("palette_representation", false, 9740u);ASSERT_TRUE(run()) << error_;
  const auto original = probe_shared_width_current();ASSERT_EQ(original.prior[8u + truth.receiver], 2u);
  const auto e = helper_.enclosure_views();const auto models = words(read(e.family_models.Get()));
  auto permuted = models;const UINT offset = 17u * truth.receiver;
  const UINT all = models[offset] & 255u, foreground = (models[offset] >> 8u) & 255u;
  UINT new_all = 0u, new_foreground = 0u;
  for (UINT mode = 0u; mode < 8u; ++mode) {
    const UINT next = (mode + 3u) % 8u;
    if ((all & (1u << mode)) != 0u) new_all |= 1u << next;
    if ((foreground & (1u << mode)) != 0u) new_foreground |= 1u << next;
    permuted[offset + 1u + 2u * next] = models[offset + 1u + 2u * mode];
    permuted[offset + 2u + 2u * next] = models[offset + 2u + 2u * mode];
  }
  permuted[offset] = new_all | (new_foreground << 8u);
  const auto reordered = probe_shared_width_model_fixture(permuted);
  EXPECT_EQ(reordered.prior, original.prior);EXPECT_EQ(reordered.candidate, original.candidate);
  // {X,X} versus {X,Y} would pass an equal-count one-way peer search. The
  // private validator must reject duplicate qualified boxes before matching.
  auto duplicates = models;
  for (UINT unit = 0u; unit < 2u; ++unit) {
    std::fill_n(duplicates.begin() + 17u * unit, 17u, 0u);duplicates[17u * unit] = 3u | (3u << 8u);
    duplicates[17u * unit + 1u] = duplicates[17u * unit + 2u] = 0x00ffffffu;
    duplicates[17u * unit + 3u] = duplicates[17u * unit + 4u] = unit == 0u ? 0x00ffffffu : 0x00808080u;
  }
  const auto rejected = probe_shared_width_model_fixture(duplicates);
  EXPECT_EQ(rejected.prior[8u], 5u);EXPECT_NE(rejected.prior[8u + truth.receiver], 2u);
  EXPECT_TRUE(std::all_of(rejected.candidate.begin(), rejected.candidate.end(), [](UINT word) { return word == 0u; }));
  RecordProperty("shared_width_palette_fixture_scope", "GPU kernel-level palette-slot permutation and malformed duplicate-box fixtures; measured production tables/resources remain unchanged");
}
