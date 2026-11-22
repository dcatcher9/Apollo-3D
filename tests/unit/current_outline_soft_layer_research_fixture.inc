// Include INSIDE HostSbsCurrentOutlineSupportClosureGpuTest, immediately after
// current_outline_unanchored_body_research_fixture.inc. No production wiring.
// body_witness_probe_t is reused ONLY as storage: rejected means recovery bits,
// filtered means actual GPU original-plus-recovery M in this separate probe.
bool soft_layer_research_configured() const {
  const auto *root = std::getenv("SUNSHINE_TEST_SOFT_LAYER_REPO_ROOT");
  const auto *pin = std::getenv("SUNSHINE_TEST_SOFT_LAYER_SHADER_PIN");
  return root && *root && pin && *pin;
}

std::array<ID3D11ShaderResourceView *, 11u> soft_layer_current_inputs() {
  const auto e = helper_.enclosure_views();const auto f = helper_.mutable_views();
  return {rgb_.srv.Get(), ocr_.srv.Get(), e.selected_bits.Get(), e.cover_bits.Get(), e.receipt.Get(),
    f.calibration_data.Get(), e.family_models.Get(), sct_.srv.Get(), f.inside_bits.Get(), e.body.Get(), f.matte.Get()};
}

body_witness_probe_t probe_soft_layer(ID3D11Buffer *constants, UINT width, UINT height,
  const std::array<ID3D11ShaderResourceView *, 11u> &inputs, const char *operand_scope, bool kernel_only_input_copies) {
  require(soft_layer_research_configured());require(constants && width && height && operand_scope);
  const auto *root = std::getenv("SUNSHINE_TEST_SOFT_LAYER_REPO_ROOT");
  const std::string expected = std::getenv("SUNSHINE_TEST_SOFT_LAYER_SHADER_PIN");
  require(expected.size() == 64u && std::all_of(expected.begin(), expected.end(), [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  }));
  const std::array specs {cache::shader_spec {"tools/sbsbench/shaders/private_current_body_soft_layer_recovery_cs.hlsl", "soft_layer_main", "cs_5_0"}};
  const auto snapshot = cache::snapshot_sources(root, specs);const auto closure = cache::source_closure_sha256(snapshot);
  require(closure == expected);const auto code = cache::get(snapshot, specs[0u]);require(code && !code->empty());
  ComPtr<ID3D11ComputeShader> shader;require(SUCCEEDED(device_->CreateComputeShader(code->data(), code->size(), nullptr, &shader)));
  std::array<std::vector<char>, 11u> before, after;
  for (UINT i = 0u; i < inputs.size(); ++i) {require(inputs[i]);before[i] = read(inputs[i]);}
  const auto cb_before = read(constants);require(cb_before.size() == 24u * sizeof(UINT));
  auto recovered = texture(width, height, DXGI_FORMAT_R32_UINT), united = texture(width, height, DXGI_FORMAT_R32_UINT), reasons = texture(width, height, DXGI_FORMAT_R32_UINT);
  auto recovery_uav = uav(recovered.srv.Get()), union_uav = uav(united.srv.Get()), reason_uav = uav(reasons.srv.Get());
  auto shape = buffer(16u, true);const std::array<UINT, 4u> dimensions {width, height, width * height, 0u};
  context_->UpdateSubresource(shape.resource.Get(), 0u, nullptr, dimensions.data(), 0u, 0u);
  ID3D11Buffer *cb[2] {constants, shape.resource.Get()};std::array<ID3D11ShaderResourceView *, 12u> in {};
  std::copy_n(inputs.begin(), 8u, in.begin());std::copy_n(inputs.begin() + 8u, 3u, in.begin() + 9u);
  std::array<ID3D11UnorderedAccessView *, 5u> out {nullptr, nullptr, recovery_uav.Get(), union_uav.Get(), reason_uav.Get()};
  context_->CSSetConstantBuffers(0u, 2u, cb);context_->CSSetShaderResources(0u, UINT(in.size()), in.data());
  context_->CSSetUnorderedAccessViews(0u, UINT(out.size()), out.data(), nullptr);
  context_->CSSetShader(shader.Get(), nullptr, 0u);context_->Dispatch((width + 15u) / 16u, (height + 15u) / 16u, 1u);
  out.fill(nullptr);in.fill(nullptr);context_->CSSetUnorderedAccessViews(0u, UINT(out.size()), out.data(), nullptr);
  context_->CSSetShaderResources(0u, UINT(in.size()), in.data());ID3D11Buffer *clear[2] {};
  context_->CSSetConstantBuffers(0u, 2u, clear);context_->CSSetShader(nullptr, nullptr, 0u);
  body_witness_probe_t result {words(read(recovered.resource.Get())), words(read(united.resource.Get())), words(read(reasons.resource.Get())), {}, {}};
  const auto count = std::size_t(width) * height;
  require(result.rejected.size() == count && result.filtered.size() == count && result.reasons.size() == count);
  for (UINT i = 0u; i < inputs.size(); ++i) after[i] = read(inputs[i]);
  const auto cb_after = read(constants);
  const bool inputs_unchanged = before == after && cb_before == cb_after;
  const bool closure_unchanged = cache::source_closure_sha256(cache::snapshot_sources(root, specs)) == closure;
  constexpr std::array<const char *, 11u> names {"CurrentRGB.rgba16f", "OCR.u32", "SelectedBits.u32", "CoverBits.u32", "CBC.u32", "F2.u32", "FamilyModels.u32", "SCT.u32", "InsideBits.u32", "BODY.u32", "FinalM.u32"};
  nlohmann::json bindings = nlohmann::json::object(), mismatches = nlohmann::json::array();
  for (UINT i = 0u; i < inputs.size(); ++i) {
    bindings[names[i]] = {{"before", {{"bytes", before[i].size()}, {"sha256", hash_bytes(before[i])}}},
      {"after", {{"bytes", after[i].size()}, {"sha256", hash_bytes(after[i])}}}, {"byteexact", before[i] == after[i]}};
    if (before[i] != after[i]) mismatches.push_back(names[i]);
    result.consumed.emplace(std::string("before_") + names[i], std::move(before[i]));
    result.consumed.emplace(std::string("after_") + names[i], std::move(after[i]));
  }
  bindings["C24.u32"] = {{"before", {{"bytes", cb_before.size()}, {"sha256", hash_bytes(cb_before)}}},
    {"after", {{"bytes", cb_after.size()}, {"sha256", hash_bytes(cb_after)}}}, {"byteexact", cb_before == cb_after}};
  if (cb_before != cb_after) mismatches.push_back("C24.u32");
  result.consumed.emplace("before_C24.u32", cb_before);result.consumed.emplace("after_C24.u32", cb_after);
  result.consumed.emplace("Shape.u32", raw(dimensions));
  const auto original = words(result.consumed.at("before_FinalM.u32"));require(original.size() == count);
  UINT proposed = 0u, additions = 0u, withdrawals = 0u, changed_without_recovery = 0u;
  for (std::size_t i = 0u; i < count; ++i) {
    proposed += result.rejected[i] != 0u;additions += original[i] == 0u && result.filtered[i] == 1u;
    withdrawals += original[i] == 1u && result.filtered[i] != 1u;
    changed_without_recovery += result.rejected[i] == 0u && result.filtered[i] != original[i];
  }
  result.metadata = {{"schema", 1u}, {"source_closure", closure}, {"source_closure_unchanged_after_probe", closure_unchanged},
    {"entry", "soft_layer_main"}, {"target", "cs_5_0"}, {"dispatches", 1u}, {"shape", {width, height}},
    {"operand_scope", operand_scope}, {"kernel_only_input_copies", kernel_only_input_copies},
    {"bound_inputs_byteexact_after_probe", inputs_unchanged}, {"bound_input_mismatches", mismatches}, {"consumed_input_bindings", bindings},
    {"legacy_storage_fields", {{"rejected", "per-unit recovery bits, NOT rejections"}, {"filtered", "actual GPU original-plus-recovery union M"}}},
    {"mask_delta", {{"proposed_recovery_pixels", proposed}, {"actual_original_M_additions", additions}, {"original_ones_withdrawn", withdrawals}, {"literal_changes_without_recovery", changed_without_recovery}}},
    {"scope", "PRIVATE single-dispatch exact same-row exterior-colour-absence hypothesis; no production mutation, alpha reconstruction or clean-background authority"},
    {"candidate_is_production_mask", false}, {"current_production_M_is_alpha_truth", false}, {"quality_accepted", false},
    {"classification_bits_are_gates", false}, {"real_development_replay_is_primary_quality_evidence", true},
    {"reason_bits", {{"original_present", 1u}, {"frame_unavailable", 2u}, {"malformed_pixel", 4u}, {"not_body", 8u},
      {"no_owned_low", 16u}, {"style_unavailable", 32u}, {"invalid_source", 64u}, {"physical_K", 128u},
      {"diagnostic_not_blacklike", 256u}, {"diagnostic_exact_foreground", 512u}, {"no_valid_exterior", 1024u},
      {"row_colour_present", 2048u}, {"recovered", 4096u}, {"malformed_row_cover", 8192u},
      {"diagnostic_blacklike", 16384u}, {"diagnostic_own_inside", 32768u}}}};
  RecordProperty("soft_layer_research_source_closure", closure);return result;
}

body_witness_probe_t probe_soft_layer_current() {
  return probe_soft_layer(cb_.resource.Get(), W_, H_, soft_layer_current_inputs(),
    "Actual unchanged current producer source/BODY/LOW/Inside/style/M operands; no copied inputs", false);
}

void export_soft_layer_probe(const char *label, const body_witness_probe_t &probe) {
  nlohmann::json report = probe.metadata;
  const auto *root = std::getenv("SUNSHINE_TEST_SOFT_LAYER_OUTPUT");
  if (root && *root) {
    const auto folder = std::filesystem::path(root) / label;require(!std::filesystem::exists(folder));require(std::filesystem::create_directories(folder));
    nlohmann::json artifacts = nlohmann::json::object();
    const auto emit = [&](const char *name, const auto &values) {
      const auto bytes = raw(values);const auto path = folder / name;std::ofstream stream(path, std::ios::binary);
      stream.write(bytes.data(), std::streamsize(bytes.size()));stream.close();require(bool(stream));
      artifacts[name] = {{"path", path.string()}, {"bytes", bytes.size()}, {"sha256", hash_bytes(bytes)}};
    };
    emit("recovery_unit_bits.u32", probe.rejected);emit("gpu_union_M.u32", probe.filtered);emit("soft_reasons.u32", probe.reasons);
    for (const auto &[name, values] : probe.consumed) emit(name.c_str(), values);
    // Context only; these are not substituted for the probe's literal operands.
    emit("context_production_M.u32", words(outline_matte()));emit("context_source_RGBA8.bin", origin_pixels_);
    report["artifacts"] = artifacts;std::ofstream stream(folder / "report.json", std::ios::binary);
    stream << report.dump(2) << '\n';stream.close();require(bool(stream));
  }
  RecordProperty(std::string(label) + "_soft_layer", report.dump());
  EXPECT_TRUE(probe.metadata.at("bound_inputs_byteexact_after_probe").get<bool>());
  EXPECT_TRUE(probe.metadata.at("source_closure_unchanged_after_probe").get<bool>());
}
