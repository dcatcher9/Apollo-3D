// Private overlap-prior owner. All source SRVs, original M and C24 are captured
// byte-exactly at learning and checked again before candidate consumption.
const char *overlap_width_root() const {
  const auto *root = std::getenv("SUNSHINE_TEST_OVERLAP_WIDTH_REPO_ROOT");return root && *root ? root : body_witness_research_root();
}
bool overlap_width_configured() const {const auto *root = overlap_width_root();return root && *root;}
constexpr static std::array<const char *, 9u> overlap_width_names {"CurrentRGB.rgba16f", "OCR.u32", "SelectedBits.u32", "CoverBits.u32", "CBC.u32", "F2.u32", "FamilyModels.u32", "SCT.u32", "OriginalM.u32"};

struct overlap_width_learned_t {
  texture_t prior;
  ComPtr<ID3D11ComputeShader> candidate_shader;
  std::vector<UINT> prior_words;
  std::map<std::string, std::vector<char>> consumed;
  std::string closure;
  nlohmann::json metadata;
};
struct overlap_width_candidate_t {
  std::vector<UINT> candidate, union_m;
  std::map<std::string, std::vector<char>> consumed;
  nlohmann::json metadata;
};

overlap_width_learned_t learn_overlap_width(ID3D11Buffer *constants, UINT width, UINT height,
  const std::array<ID3D11ShaderResourceView *, 9u> &inputs) {
  require(overlap_width_configured());overlap_width_learned_t result;
  for (UINT i = 0u; i < inputs.size(); ++i) {require(inputs[i]);result.consumed.emplace(overlap_width_names[i], read(inputs[i]));}
  result.consumed.emplace("Constants.u32", read(constants));
  const std::array specs {cache::shader_spec {"tools/sbsbench/shaders/host_sbs_shared_width_overlap_prior_research_cs.hlsl", "overlap_prior_main", "cs_5_0"},
    cache::shader_spec {"tools/sbsbench/shaders/host_sbs_shared_width_overlap_prior_research_cs.hlsl", "overlap_candidate_main", "cs_5_0"}};
  const auto snapshot = cache::snapshot_sources(overlap_width_root(), specs);result.closure = cache::source_closure_sha256(snapshot);
  if (const auto *pin = std::getenv("SUNSHINE_TEST_OVERLAP_WIDTH_SHADER_PIN")) require(result.closure == pin);
  std::array<ComPtr<ID3D11ComputeShader>, 2u> shaders;
  for (UINT i = 0u; i < shaders.size(); ++i) {const auto code = cache::get(snapshot, specs[i]);require(code && !code->empty());require(SUCCEEDED(device_->CreateComputeShader(code->data(), code->size(), nullptr, &shaders[i])));}
  result.candidate_shader = shaders[1u];result.prior = texture(8u, 20u, DXGI_FORMAT_R32_UINT);auto pu = uav(result.prior.srv.Get());auto shape = buffer(16u, true);
  const std::array<UINT, 4u> dimensions {width, height, width * height, 0u};context_->UpdateSubresource(shape.resource.Get(), 0u, nullptr, dimensions.data(), 0u, 0u);
  std::array<ID3D11ShaderResourceView *, 10u> in {};std::copy_n(inputs.begin(), 8u, in.begin());in[9u] = inputs[8u];
  std::array<ID3D11UnorderedAccessView *, 3u> out {pu.Get(), nullptr, nullptr};ID3D11Buffer *cb[2] {constants, shape.resource.Get()};
  context_->CSSetConstantBuffers(0u, 2u, cb);context_->CSSetShaderResources(0u, UINT(in.size()), in.data());context_->CSSetUnorderedAccessViews(0u, UINT(out.size()), out.data(), nullptr);
  context_->CSSetShader(shaders[0u].Get(), nullptr, 0u);context_->Dispatch(1u, 1u, 1u);
  in.fill(nullptr);out.fill(nullptr);context_->CSSetUnorderedAccessViews(0u, UINT(out.size()), out.data(), nullptr);context_->CSSetShaderResources(0u, UINT(in.size()), in.data());
  ID3D11Buffer *clear[2] {};context_->CSSetConstantBuffers(0u, 2u, clear);context_->CSSetShader(nullptr, nullptr, 0u);
  result.prior_words = words(read(result.prior.resource.Get()));require(result.prior_words.size() == 160u);
  nlohmann::json bindings = nlohmann::json::object();for (const auto &[name, bytes] : result.consumed) bindings[name] = {{"bytes", bytes.size()}, {"sha256", hash_bytes(bytes)}};
  for (UINT i = 0u; i < inputs.size(); ++i) require(read(inputs[i]) == result.consumed.at(overlap_width_names[i]));require(read(constants) == result.consumed.at("Constants.u32"));
  result.metadata = {{"schema", 1u}, {"source_closure", result.closure}, {"consumed_input_bindings", bindings}, {"production_inputs_byteexact_after_learning", true},
    {"prior_shape", {8u, 20u}}, {"shape", {width, height}}, {"scope", "PRIVATE overlap-style prior; strong unanimous current donors; borrowed width never mutates local KNOWN/F2"},
    {"prior_statuses", {{"unavailable", 0u}, {"local_strong", 1u}, {"borrowed", 2u}, {"conflicting_widths", 3u}, {"no_donor", 4u}, {"no_style", 5u}, {"no_fill", 6u}, {"bad_receiver", 7u}, {"ambiguous_families", 8u}, {"local_conflict", 9u}}},
    {"pair_rejection_bits", {{"not_strong", 1u}, {"recipient_geometry", 2u}, {"disjoint_K", 4u}, {"unmatched_FG", 8u}, {"self", 16u}, {"receiver_unavailable", 32u}, {"ambiguous_FG_overlap", 64u}}},
    {"candidate_is_alpha_truth", false}, {"quality_accepted", false}};
  return result;
}

overlap_width_candidate_t consume_overlap_width(ID3D11Buffer *constants, UINT width, UINT height,
  const std::array<ID3D11ShaderResourceView *, 9u> &inputs, const overlap_width_learned_t &learned) {
  require(overlap_width_configured());overlap_width_candidate_t result;bool source_matches = true;nlohmann::json mismatches = nlohmann::json::array();
  for (UINT i = 0u; i < inputs.size(); ++i) {require(inputs[i]);result.consumed.emplace(overlap_width_names[i], read(inputs[i]));}
  result.consumed.emplace("Constants.u32", read(constants));
  for (const auto &[name, bytes] : result.consumed) {const auto found = learned.consumed.find(name);if (found == learned.consumed.end() || found->second != bytes) {source_matches = false;mismatches.push_back(name);}}
  const auto prior_before = read(learned.prior.resource.Get());const bool table_matches = prior_before == raw(learned.prior_words);
  const std::array specs {cache::shader_spec {"tools/sbsbench/shaders/host_sbs_shared_width_overlap_prior_research_cs.hlsl", "overlap_prior_main", "cs_5_0"},
    cache::shader_spec {"tools/sbsbench/shaders/host_sbs_shared_width_overlap_prior_research_cs.hlsl", "overlap_candidate_main", "cs_5_0"}};
  const auto snapshot = cache::snapshot_sources(overlap_width_root(), specs);const auto closure = cache::source_closure_sha256(snapshot);const bool policy_matches = closure == learned.closure;
  if (const auto *pin = std::getenv("SUNSHINE_TEST_OVERLAP_WIDTH_SHADER_PIN")) require(closure == pin);
  const bool usable = source_matches && table_matches && policy_matches;const std::array<UINT, 4u> gate {0x3157504fu, 1u, usable ? 1u : 0u, 0u};
  auto candidate = texture(width, height, DXGI_FORMAT_R32_UINT), union_m = texture(width, height, DXGI_FORMAT_R32_UINT);auto cu = uav(candidate.srv.Get()), uu = uav(union_m.srv.Get());
  const std::array<UINT, 4u> zero {};context_->ClearUnorderedAccessViewUint(cu.Get(), zero.data());
  // Literal fallback and every scored union are actual GPU resource operations.
  // No CPU mask OR or alpha-derived candidate is constructed, even on mismatch.
  ComPtr<ID3D11Resource> original;inputs[8u]->GetResource(&original);D3D11_RESOURCE_DIMENSION type {};original->GetType(&type);require(type == D3D11_RESOURCE_DIMENSION_TEXTURE2D);
  ComPtr<ID3D11Texture2D> original_texture;require(SUCCEEDED(original.As(&original_texture)));D3D11_TEXTURE2D_DESC desc {};original_texture->GetDesc(&desc);require(desc.Width == width && desc.Height == height && desc.Format == DXGI_FORMAT_R32_UINT);
  context_->CopyResource(union_m.resource.Get(), original.Get());
  if (usable) {
    require(learned.candidate_shader.Get() != nullptr);auto shape = buffer(16u, true);const std::array<UINT, 4u> dimensions {width, height, width * height, 0u};context_->UpdateSubresource(shape.resource.Get(), 0u, nullptr, dimensions.data(), 0u, 0u);
    std::array<ID3D11ShaderResourceView *, 10u> in {};std::copy_n(inputs.begin(), 8u, in.begin());in[8u] = learned.prior.srv.Get();in[9u] = inputs[8u];
    std::array<ID3D11UnorderedAccessView *, 3u> out {nullptr, cu.Get(), uu.Get()};ID3D11Buffer *cb[2] {constants, shape.resource.Get()};context_->CSSetConstantBuffers(0u, 2u, cb);
    context_->CSSetShaderResources(0u, UINT(in.size()), in.data());context_->CSSetUnorderedAccessViews(0u, UINT(out.size()), out.data(), nullptr);context_->CSSetShader(learned.candidate_shader.Get(), nullptr, 0u);context_->Dispatch((width + 15u) / 16u, (height + 15u) / 16u, 1u);
    in.fill(nullptr);out.fill(nullptr);context_->CSSetUnorderedAccessViews(0u, UINT(out.size()), out.data(), nullptr);context_->CSSetShaderResources(0u, UINT(in.size()), in.data());ID3D11Buffer *clear[2] {};context_->CSSetConstantBuffers(0u, 2u, clear);context_->CSSetShader(nullptr, nullptr, 0u);
  }
  result.candidate = words(read(candidate.resource.Get()));result.union_m = words(read(union_m.resource.Get()));require(result.candidate.size() == std::size_t(width) * height && result.union_m.size() == result.candidate.size());
  for (UINT i = 0u; i < inputs.size(); ++i) require(read(inputs[i]) == result.consumed.at(overlap_width_names[i]));require(read(constants) == result.consumed.at("Constants.u32"));require(read(learned.prior.resource.Get()) == prior_before);
  result.consumed.emplace("Prior.u32", prior_before);result.consumed.emplace("ConsumptionGate.u32", raw(gate));nlohmann::json bindings = nlohmann::json::object();for (const auto &[name, bytes] : result.consumed) bindings[name] = {{"bytes", bytes.size()}, {"sha256", hash_bytes(bytes)}};
  result.metadata = {{"schema", 1u}, {"source_closure", closure}, {"learning", learned.metadata}, {"source_tuple_matches_learning", source_matches}, {"source_tuple_mismatches", mismatches}, {"prior_bytes_match_learning", table_matches}, {"shader_source_matches_learning", policy_matches},
    {"candidate_dispatched", usable}, {"consumption_gate", gate}, {"source_inputs_and_prior_byteexact_after_consumption", true}, {"consumed_input_bindings", bindings},
    {"union_M_origin", "GPU CopyResource of CURRENT original M, then optional actual overlap_candidate_main writes"}, {"CPU_union_constructed", false}, {"candidate_is_production_mask", false}, {"quality_accepted", false}};
  return result;
}

std::array<ID3D11ShaderResourceView *, 9u> current_overlap_width_inputs(ID3D11ShaderResourceView *rgb = nullptr, ID3D11ShaderResourceView *family = nullptr, ID3D11ShaderResourceView *original_m = nullptr) {
  const auto e = helper_.enclosure_views();const auto f = helper_.mutable_views();return {rgb ? rgb : rgb_.srv.Get(), ocr_.srv.Get(), e.selected_bits.Get(), e.cover_bits.Get(), e.receipt.Get(), f.calibration_data.Get(), family ? family : e.family_models.Get(), sct_.srv.Get(), original_m ? original_m : f.matte.Get()};
}

shared_width_authored_t author_overlap_width_case(bool conflict, UINT frame) {
  auto truth = author_shared_width_case(conflict ? "mixed_widths" : "matching", false, frame);const UINT left = truth.receiver_left;
  // Independent original layers: nonzero literal-dark scene makes true black
  // ink visible by one RGB8 code and prevents the receiver's local K exit.
  // This faint authored positive is mechanics evidence, not a >=3-code gain.
  for (UINT y = 16u; y < 112u; ++y) for (UINT x = left + 16u; x < left + 240u; ++x) truth.scene[y * W_ + x] = {1u, 1u, 1u, 255u};
  origin_pixels_ = truth.scene;for (std::size_t i = 0u; i < origin_pixels_.size(); ++i) for (UINT c = 0u; c < 3u; ++c) {const UINT alpha = truth.alpha[i];origin_pixels_[i][c] = std::uint8_t((alpha * truth.ink[i][c] + (255u - alpha) * truth.scene[i][c] + 127u) / 255u);}
  ocr_words_[16u + 8u * truth.receiver + 1u] = 6u; // Different observed core height, same authored glyph raster.
  truth.source_identical_ambiguity = false;upload_pixels_from_origin();return truth;
}

void export_overlap_width_probe(const char *label, const overlap_width_learned_t &learned, const overlap_width_candidate_t &candidate,
  const shared_width_authored_t *truth, const char *scope) {
  auto report = candidate.metadata;report["learning"] = learned.metadata;report["fixture_scope"] = scope;
  if (truth) {
    UINT added_ink = 0u, added_scene = 0u, visible_ink = 0u, strong_ink = 0u, withdrawn_ink = 0u;const auto original = words(candidate.consumed.at("OriginalM.u32"));
    for (std::size_t i = 0u; i < candidate.union_m.size(); ++i) {
      const bool added = original[i] == 0u && candidate.union_m[i] == 1u, ink = truth->alpha[i] != 0u;UINT delta = 0u;
      for (UINT c = 0u; c < 3u; ++c) delta = std::max(delta, UINT(std::abs(int(origin_pixels_[i][c]) - int(truth->scene[i][c]))));
      added_ink += added && ink;added_scene += added && !ink;visible_ink += added && ink && delta >= 1u;strong_ink += added && ink && truth->alpha[i] >= 5u && delta >= 3u;withdrawn_ink += ink && original[i] != 0u && candidate.union_m[i] == 0u;
    }
    report["independent_authored_truth"] = {{"added_ink", added_ink}, {"added_scene", added_scene}, {"added_visible_ink_delta_ge1", visible_ink}, {"added_strong_ink_delta_ge3", strong_ink}, {"withdrawn_ink", withdrawn_ink}, {"authoring", "Independent original scene/ink/alpha; no truth supplied to GPU; darkRGB1 recipient gives faint one-code original-black contrast"}};
  }
  if (const auto *root = std::getenv("SUNSHINE_TEST_OVERLAP_WIDTH_OUTPUT")) {
    const auto folder = std::filesystem::path(root) / label;require(!std::filesystem::exists(folder));require(std::filesystem::create_directories(folder));nlohmann::json artifacts = nlohmann::json::object();
    const auto emit = [&](const std::string &name, const std::vector<char> &bytes) {const auto path = folder / name;std::ofstream stream(path, std::ios::binary);stream.write(bytes.data(), std::streamsize(bytes.size()));stream.close();require(bool(stream));artifacts[name] = {{"path", path.string()}, {"bytes", bytes.size()}, {"sha256", hash_bytes(bytes)}};};
    for (const auto &[name, bytes] : learned.consumed) emit("learn_consumed_" + name, bytes);for (const auto &[name, bytes] : candidate.consumed) emit("candidate_consumed_" + name, bytes);
    emit("prior.u32", raw(learned.prior_words));emit("candidate.u32", raw(candidate.candidate));emit("union_M.u32", raw(candidate.union_m));
    if (truth) {emit("authored_scene.rgba8", raw(truth->scene));emit("authored_ink.rgba8", raw(truth->ink));emit("authored_alpha.u8", raw(truth->alpha));emit("authored_source.rgba8", raw(origin_pixels_));}
    report["artifacts"] = artifacts;std::ofstream stream(folder / "report.json", std::ios::binary);stream << report.dump(2) << '\n';stream.close();require(bool(stream));
  }
  RecordProperty(std::string(label) + "_overlap_width", report.dump());
}
