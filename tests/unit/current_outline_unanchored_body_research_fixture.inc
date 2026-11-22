// Actual existing Inside graph observer; private outputs never feed the host.
struct body_witness_probe_t {
  std::vector<UINT> rejected, filtered, reasons;
  nlohmann::json metadata;
  std::map<std::string, std::vector<char>> consumed;
};

const char *body_witness_research_root() const {
  const auto *root = std::getenv("SUNSHINE_TEST_BODY_WITNESS_REPO_ROOT");
  if (root && *root) return root;
  return std::getenv("SUNSHINE_TEST_SHARED_WIDTH_REPO_ROOT");
}

bool body_witness_research_configured() const {
  const auto *root = body_witness_research_root();return root && *root;
}

body_witness_probe_t probe_body_witness(ID3D11Buffer *constants, UINT width, UINT height,
  const std::array<ID3D11ShaderResourceView *, 11u> &inputs) {
  require(body_witness_research_configured());
  const std::array specs {cache::shader_spec {"tools/sbsbench/shaders/host_sbs_unanchored_body_research_cs.hlsl", "unanchored_body_main", "cs_5_0"}};
  const auto snapshot = cache::snapshot_sources(body_witness_research_root(), specs);
  const auto closure = cache::source_closure_sha256(snapshot);
  if (const auto *pin = std::getenv("SUNSHINE_TEST_BODY_WITNESS_SHADER_PIN")) require(closure == pin);
  const auto code = cache::get(snapshot, specs[0u]);require(code && !code->empty());
  ComPtr<ID3D11ComputeShader> shader;require(SUCCEEDED(device_->CreateComputeShader(code->data(), code->size(), nullptr, &shader)));
  std::array<std::vector<char>, 11u> before;
  for (UINT i = 0u; i < inputs.size(); ++i) {require(inputs[i] != nullptr);before[i] = read(inputs[i]);}
  const auto cb_before = read(constants);
  auto rejected = texture(width, height, DXGI_FORMAT_R32_UINT), filtered = texture(width, height, DXGI_FORMAT_R32_UINT), reasons = texture(width, height, DXGI_FORMAT_R32_UINT);
  auto rejected_uav = uav(rejected.srv.Get()), filtered_uav = uav(filtered.srv.Get()), reasons_uav = uav(reasons.srv.Get());
  auto shape = buffer(16u, true);const std::array<UINT, 4u> dimensions {width, height, width * height, 0u};
  context_->UpdateSubresource(shape.resource.Get(), 0u, nullptr, dimensions.data(), 0u, 0u);
  ID3D11Buffer *cb[2] {constants, shape.resource.Get()};std::array<ID3D11ShaderResourceView *, 12u> in {};
  std::copy_n(inputs.begin(), 8u, in.begin());std::copy_n(inputs.begin() + 8u, 3u, in.begin() + 9u);
  std::array<ID3D11UnorderedAccessView *, 5u> out {nullptr, nullptr, rejected_uav.Get(), filtered_uav.Get(), reasons_uav.Get()};
  context_->CSSetConstantBuffers(0u, 2u, cb);context_->CSSetShaderResources(0u, UINT(in.size()), in.data());
  context_->CSSetUnorderedAccessViews(0u, UINT(out.size()), out.data(), nullptr);
  context_->CSSetShader(shader.Get(), nullptr, 0u);context_->Dispatch((width + 15u) / 16u, (height + 15u) / 16u, 1u);
  out.fill(nullptr);in.fill(nullptr);context_->CSSetUnorderedAccessViews(0u, UINT(out.size()), out.data(), nullptr);
  context_->CSSetShaderResources(0u, UINT(in.size()), in.data());ID3D11Buffer *clear[2] {};
  context_->CSSetConstantBuffers(0u, 2u, clear);context_->CSSetShader(nullptr, nullptr, 0u);
  body_witness_probe_t result {words(read(rejected.resource.Get())), words(read(filtered.resource.Get())), words(read(reasons.resource.Get())), {}, {}};
  const auto pixels = std::size_t(width) * height;
  require(result.rejected.size() == pixels && result.filtered.size() == pixels && result.reasons.size() == pixels);
  for (UINT i = 0u; i < inputs.size(); ++i) require(read(inputs[i]) == before[i]);require(read(constants) == cb_before);
  result.metadata = {{"schema", 1u}, {"source_closure", closure}, {"production_inputs_byteexact_after_probe", true}, {"shape", {width, height}},
    {"scope", "PRIVATE existing-Inside/current-family witness observer; no production mask mutation, new graph, new RGB threshold or routing"},
    {"rejection_permission", "Current BODY+owned LOW; every owner current style/Inside usable, nonblacklike, outside exact qualified FG, ownInside absent; no covered valid other-unit keep"},
    {"high_bits_are_independent_foreground_evidence", false}, {"current_production_M_is_alpha_truth", false},
    {"candidate_is_production_mask", false}, {"quality_accepted", false},
    {"reason_bits", {{"body", 1u}, {"keep_unknown", 2u}, {"keep_exact_foreground", 4u}, {"keep_existing_inside", 8u},
      {"keep_blacklike", 16u}, {"keep_other_unit", 32u}, {"reject", 64u}, {"original_high_present", 128u}, {"other_covered_unit_present", 256u}}}};
  constexpr std::array<const char *, 11u> names {"CurrentRGB.rgba16f", "OCR.u32", "SelectedBits.u32", "CoverBits.u32", "CBC.u32", "F2.u32", "FamilyModels.u32", "SCT.u32", "InsideBits.u32", "BODY.u32", "FinalM.u32"};
  nlohmann::json bindings = nlohmann::json::object();
  for (UINT i = 0u; i < inputs.size(); ++i) {
    bindings[names[i]] = {{"bytes", before[i].size()}, {"sha256", hash_bytes(before[i])}};
    result.consumed.emplace(names[i], std::move(before[i]));
  }
  result.consumed.emplace("Constants.u32", cb_before);bindings["Constants.u32"] = {{"bytes", cb_before.size()}, {"sha256", hash_bytes(cb_before)}};
  result.metadata["consumed_input_bindings"] = std::move(bindings);
  const auto original_words = words(result.consumed.at("FinalM.u32"));UINT proposed = 0u, withdrawn = 0u, added = 0u;
  for (std::size_t i = 0u; i < pixels; ++i) {
    proposed += result.rejected[i] != 0u;withdrawn += original_words[i] == 1u && result.filtered[i] == 0u;
    added += original_words[i] == 0u && result.filtered[i] == 1u;
  }
  result.metadata["mask_delta"] = {{"proposed_rejected_body_pixels", proposed}, {"actual_M_withdrawals", withdrawn}, {"actual_M_additions", added}};
  RecordProperty("unanchored_body_research_source_closure", closure);return result;
}

body_witness_probe_t probe_body_witness_current(ID3D11ShaderResourceView *inside_override = nullptr,
  ID3D11ShaderResourceView *selected_override = nullptr) {
  const auto e = helper_.enclosure_views();const auto f = helper_.mutable_views();
  return probe_body_witness(cb_.resource.Get(), W_, H_, {rgb_.srv.Get(), ocr_.srv.Get(), selected_override ? selected_override : e.selected_bits.Get(),
    e.cover_bits.Get(), e.receipt.Get(), f.calibration_data.Get(), e.family_models.Get(), sct_.srv.Get(),
    inside_override ? inside_override : f.inside_bits.Get(), e.body.Get(), f.matte.Get()});
}

void export_body_witness_probe(const char *label, const body_witness_probe_t &probe, const char *scope) {
  nlohmann::json report = probe.metadata;report["fixture_scope"] = scope;
  UINT rejects = 0u;for (const UINT word : probe.rejected) rejects += word != 0u;
  report["proposed_rejected_body_pixels"] = rejects;
  const auto *root = std::getenv("SUNSHINE_TEST_BODY_WITNESS_OUTPUT");
  if (root && *root) {
    const auto folder = std::filesystem::path(root) / label;require(!std::filesystem::exists(folder));require(std::filesystem::create_directories(folder));
    nlohmann::json artifacts = nlohmann::json::object();
    const auto emit = [&](const char *name, const auto &values) {
      const auto bytes = raw(values);const auto path = folder / name;std::ofstream stream(path, std::ios::binary);
      stream.write(bytes.data(), std::streamsize(bytes.size()));stream.close();require(bool(stream));
      artifacts[name] = {{"path", path.string()}, {"bytes", bytes.size()}, {"sha256", hash_bytes(bytes)}};
    };
    emit("rejected.u32", probe.rejected);emit("filtered_M.u32", probe.filtered);emit("keep_reasons.u32", probe.reasons);
    emit("production_M.u32", words(outline_matte()));emit("production_BODY.u32", words(body()));
    emit("production_InsideBits.u32", words(read(helper_.mutable_views().inside_bits.Get())));emit("production_SelectedBits.u32", words(read(helper_.enclosure_views().selected_bits.Get())));
    for (const auto &[name, bytes] : probe.consumed) emit(("consumed_" + name).c_str(), bytes);
    emit("source_RGBA8.bin", origin_pixels_);emit("C24.u32", cb_words_);emit("SCT.u32", sct_words_);
    report["artifacts"] = artifacts;std::ofstream stream(folder / "report.json", std::ios::binary);stream << report.dump(2) << '\n';stream.close();require(bool(stream));
  }
  RecordProperty(std::string(label) + "_body_witness", report.dump());
}
