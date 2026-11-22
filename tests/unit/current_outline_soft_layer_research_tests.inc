// Include OUTSIDE the class, after current_outline_unanchored_body_research_tests.inc.
// Copied M/source operands below are explicitly kernel-only. None proves that a
// real producer omitted a soft pixel, nor that a proposed addition is text alpha.
TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, SoftLayerResearchActualCurrentInputsPreserveOriginalOnes) {
  if (!soft_layer_research_configured()) GTEST_SKIP() << "Explicit private soft-layer repository root and closure pin required";
  author_band(5u, 9970u);ASSERT_TRUE(run()) << error_;
  const auto original = words(outline_matte());const auto before_body = body();
  const auto probe = probe_soft_layer_current();export_soft_layer_probe("soft_actual_current", probe);
  ASSERT_TRUE(probe.metadata.at("bound_inputs_byteexact_after_probe").get<bool>());
  ASSERT_TRUE(probe.metadata.at("source_closure_unchanged_after_probe").get<bool>());
  EXPECT_EQ(body(), before_body);EXPECT_EQ(words(outline_matte()), original);
  const auto current_body = words(probe.consumed.at("before_BODY.u32"));
  const auto low = words(probe.consumed.at("before_SelectedBits.u32"));
  for (std::size_t i = 0u; i < original.size(); ++i) {
    if (original[i] == 1u) {EXPECT_EQ(probe.filtered[i], 1u);EXPECT_EQ(probe.rejected[i], 0u);}
    if (probe.rejected[i] != 0u) {
      EXPECT_EQ(original[i], 0u);EXPECT_EQ(current_body[i], 1u);EXPECT_EQ(probe.rejected[i] & ~(low[i] & 255u), 0u);
      EXPECT_EQ(probe.filtered[i], 1u);EXPECT_NE(probe.reasons[i] & 4096u, 0u);
    }
  }
  EXPECT_EQ(probe.metadata.at("mask_delta").at("original_ones_withdrawn"), 0u);
  EXPECT_EQ(probe.metadata.at("mask_delta").at("literal_changes_without_recovery"), 0u);
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, SoftLayerResearchKernelOnlyCopiedMissingMAndRowVeto) {
  if (!soft_layer_research_configured()) GTEST_SKIP() << "Explicit private soft-layer repository root and closure pin required";
  author_band(5u, 9971u);ASSERT_TRUE(run()) << error_;
  constexpr UINT x = 90u, y = 60u;const UINT q = y * W_ + x, exterior = y * W_;
  auto inputs = soft_layer_current_inputs();const auto original = words(read(inputs[10u]));
  const auto current_body = words(read(inputs[9u])), low = words(read(inputs[2u])), cover = words(read(inputs[3u]));
  ASSERT_EQ(original[q], 1u);ASSERT_EQ(current_body[q], 1u);ASSERT_NE(low[q] & 1u, 0u);ASSERT_EQ(cover[exterior], 0u);
  auto copied_m = original;copied_m[q] = 0u;auto m = texture(W_, H_, DXGI_FORMAT_R32_UINT);
  context_->UpdateSubresource(m.resource.Get(), 0u, nullptr, copied_m.data(), W_ * 4u, 0u);inputs[10u] = m.srv.Get();
  auto source = read(inputs[0u]);const pixel16 soft {code_half(160u), code_half(160u), code_half(160u), code_half(255u)};
  std::memcpy(source.data() + std::size_t(q) * 8u, soft.data(), sizeof(soft));
  auto rgb = texture(W_, H_, DXGI_FORMAT_R16G16B16A16_FLOAT);
  context_->UpdateSubresource(rgb.resource.Get(), 0u, nullptr, source.data(), W_ * 8u, 0u);inputs[0u] = rgb.srv.Get();
  const auto recovered = probe_soft_layer(cb_.resource.Get(), W_, H_, inputs,
    "KERNEL-ONLY copied original M clears one real protected BODY/LOW texel; copied source changes only that texel to gray160", true);
  export_soft_layer_probe("soft_kernel_unique", recovered);
  ASSERT_TRUE(recovered.metadata.at("bound_inputs_byteexact_after_probe").get<bool>());
  EXPECT_EQ(recovered.rejected[q], 1u);EXPECT_EQ(recovered.filtered[q], 1u);EXPECT_NE(recovered.reasons[q] & 4096u, 0u);
  // Gray160 is nearer the current white family than K0. This proves the
  // existing classification is reported, rather than silently made a gate.
  EXPECT_NE(recovered.reasons[q] & 256u, 0u);
  std::memcpy(source.data() + std::size_t(exterior) * 8u, soft.data(), sizeof(soft));
  context_->UpdateSubresource(rgb.resource.Get(), 0u, nullptr, source.data(), W_ * 8u, 0u);
  const auto matched = probe_soft_layer(cb_.resource.Get(), W_, H_, inputs,
    "KERNEL-ONLY same copied missing-M/source pair plus one exact gray160 exterior texel on the target native row", true);
  export_soft_layer_probe("soft_kernel_row_match", matched);
  EXPECT_EQ(matched.rejected[q], 0u);EXPECT_EQ(matched.filtered[q], 0u);EXPECT_NE(matched.reasons[q] & 2048u, 0u);
  auto all_cover = cover;for (UINT px = 0u; px < W_; ++px) all_cover[y * W_ + px] = 1u;
  auto cover_copy = texture(W_, H_, DXGI_FORMAT_R32_UINT);
  context_->UpdateSubresource(cover_copy.resource.Get(), 0u, nullptr, all_cover.data(), W_ * 4u, 0u);inputs[3u] = cover_copy.srv.Get();
  const auto no_pool = probe_soft_layer(cb_.resource.Get(), W_, H_, inputs,
    "KERNEL-ONLY cover copy marks every texel on the target row covered, so no valid exterior pool exists", true);
  export_soft_layer_probe("soft_kernel_no_exterior", no_pool);
  EXPECT_EQ(no_pool.rejected[q], 0u);EXPECT_EQ(no_pool.filtered[q], 0u);EXPECT_NE(no_pool.reasons[q] & 1024u, 0u);
  EXPECT_EQ(words(outline_matte()), original);
}

TEST_F(HostSbsCurrentOutlineSupportClosureGpuTest, SoftLayerResearchKernelOnlyPhysicalUnknownAndMalformedAbstain) {
  if (!soft_layer_research_configured()) GTEST_SKIP() << "Explicit private soft-layer repository root and closure pin required";
  author_band(5u, 9972u);ASSERT_TRUE(run()) << error_;const UINT q = 60u * W_ + 90u;
  auto inputs = soft_layer_current_inputs();const auto original = words(read(inputs[10u]));ASSERT_EQ(original[q], 1u);
  auto copied_m = original;copied_m[q] = 0u;auto m = texture(W_, H_, DXGI_FORMAT_R32_UINT);
  context_->UpdateSubresource(m.resource.Get(), 0u, nullptr, copied_m.data(), W_ * 4u, 0u);inputs[10u] = m.srv.Get();
  auto source = read(inputs[0u]);const pixel16 black {0u, 0u, 0u, code_half(255u)};
  std::memcpy(source.data() + std::size_t(q) * 8u, black.data(), sizeof(black));auto rgb = texture(W_, H_, DXGI_FORMAT_R16G16B16A16_FLOAT);
  context_->UpdateSubresource(rgb.resource.Get(), 0u, nullptr, source.data(), W_ * 8u, 0u);inputs[0u] = rgb.srv.Get();
  const auto physical = probe_soft_layer(cb_.resource.Get(), W_, H_, inputs,
    "KERNEL-ONLY copied missing M and physical-black source at a real BODY/LOW point; this experiment must not recover K", true);
  export_soft_layer_probe("soft_kernel_physical", physical);
  EXPECT_EQ(physical.rejected[q], 0u);EXPECT_EQ(physical.filtered[q], 0u);EXPECT_NE(physical.reasons[q] & 128u, 0u);
  inputs[0u] = rgb_.srv.Get();
  auto constants = cb_words_;constants[12u] ^= 1u;auto stale_cb = buffer(96u, true);
  context_->UpdateSubresource(stale_cb.resource.Get(), 0u, nullptr, constants.data(), 0u, 0u);
  const auto stale = probe_soft_layer(stale_cb.resource.Get(), W_, H_, inputs,
    "KERNEL-ONLY copied missing M plus stale C24; all publication must retain that literal input M", true);
  export_soft_layer_probe("soft_kernel_stale", stale);
  EXPECT_EQ(stale.filtered, copied_m);EXPECT_TRUE(std::all_of(stale.rejected.begin(), stale.rejected.end(), [](UINT v) { return v == 0u; }));
  auto unavailable = texture(1u, 1u, DXGI_FORMAT_R32_UINT);const UINT zero = 0u;
  context_->UpdateSubresource(unavailable.resource.Get(), 0u, nullptr, &zero, sizeof(zero), 0u);inputs[8u] = unavailable.srv.Get();
  const auto missing = probe_soft_layer(cb_.resource.Get(), W_, H_, inputs,
    "KERNEL-ONLY copied missing M plus wrong-shape current Inside; literal copied M must remain unchanged", true);
  export_soft_layer_probe("soft_kernel_unavailable_inside", missing);
  EXPECT_EQ(missing.filtered, copied_m);EXPECT_TRUE(std::all_of(missing.rejected.begin(), missing.rejected.end(), [](UINT v) { return v == 0u; }));
  inputs[8u] = helper_.mutable_views().inside_bits.Get();
  auto cbc = words(read(inputs[4u]));cbc[128u + 6u] = 2u;auto unknown = buffer(512u * 4u);
  context_->UpdateSubresource(unknown.resource.Get(), 0u, nullptr, cbc.data(), 0u, 0u);inputs[4u] = unknown.srv.Get();
  const auto unadopted = probe_soft_layer(cb_.resource.Get(), W_, H_, inputs,
    "KERNEL-ONLY copied missing M plus copied CBC selectionStatus removes physical adoption; no soft recovery permission", true);
  export_soft_layer_probe("soft_kernel_unadopted", unadopted);
  EXPECT_EQ(unadopted.rejected[q], 0u);EXPECT_EQ(unadopted.filtered[q], 0u);EXPECT_NE(unadopted.reasons[q] & 32u, 0u);
  inputs[4u] = helper_.enclosure_views().receipt.Get();copied_m[q] = 7u;
  context_->UpdateSubresource(m.resource.Get(), 0u, nullptr, copied_m.data(), W_ * 4u, 0u);
  const auto noncanonical = probe_soft_layer(cb_.resource.Get(), W_, H_, inputs,
    "KERNEL-ONLY copied nonbinary original M word7; malformed input must be preserved literally, not canonicalized or recovered", true);
  export_soft_layer_probe("soft_kernel_nonbinary_original", noncanonical);
  EXPECT_EQ(noncanonical.rejected[q], 0u);EXPECT_EQ(noncanonical.filtered[q], 7u);EXPECT_NE(noncanonical.reasons[q] & 4u, 0u);
  EXPECT_EQ(words(outline_matte()), original);
}
