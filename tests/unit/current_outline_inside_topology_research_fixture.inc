// Test-only replay of the actual canonical Inside/ROI shaders in fresh scratch.
// Existing InsideBits parity is a mandatory gate, not a semantic-alpha verdict.
struct topology_words_t {
  ComPtr<ID3D11Buffer> resource;
  ComPtr<ID3D11ShaderResourceView> srv;
  ComPtr<ID3D11UnorderedAccessView> uav;
};

topology_words_t topology_words(UINT count, bool raw_view = false, bool indirect = false) {
  topology_words_t result;D3D11_BUFFER_DESC desc {};
  desc.ByteWidth = count * 4u;desc.Usage = D3D11_USAGE_DEFAULT;
  desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
  desc.MiscFlags = raw_view ? D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS : D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
  if (indirect) desc.MiscFlags |= D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
  desc.StructureByteStride = raw_view ? 0u : 4u;
  require(SUCCEEDED(device_->CreateBuffer(&desc, nullptr, &result.resource)));
  D3D11_SHADER_RESOURCE_VIEW_DESC sr {};
  sr.Format = raw_view ? DXGI_FORMAT_R32_TYPELESS : DXGI_FORMAT_UNKNOWN;
  sr.ViewDimension = raw_view ? D3D11_SRV_DIMENSION_BUFFEREX : D3D11_SRV_DIMENSION_BUFFER;
  if (raw_view) {sr.BufferEx.NumElements = count;sr.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;}
  else sr.Buffer.NumElements = count;
  D3D11_UNORDERED_ACCESS_VIEW_DESC ur {};ur.Format = sr.Format;ur.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
  ur.Buffer.NumElements = count;ur.Buffer.Flags = raw_view ? D3D11_BUFFER_UAV_FLAG_RAW : 0u;
  require(SUCCEEDED(device_->CreateShaderResourceView(result.resource.Get(), &sr, &result.srv)));
  require(SUCCEEDED(device_->CreateUnorderedAccessView(result.resource.Get(), &ur, &result.uav)));return result;
}

struct inside_topology_probe_t {
  texture_t closed, exterior, seeded, unknown, replay_inside;
  topology_words_t header;
  std::vector<UINT> replay_words, header_words;
  std::map<std::string, std::vector<char>> artifacts;
  nlohmann::json metadata;
  bool byteexact = false;
};

inside_topology_probe_t replay_inside_topology(ID3D11Buffer *constants, UINT width, UINT height,
  const std::array<ID3D11ShaderResourceView *, 11u> &body_inputs,
  ID3D11ShaderResourceView *all_body_bits, ID3D11ShaderResourceView *raw_m, ID3D11ShaderResourceView *code1_m,
  const std::array<ID3D11ComputeShader *, 4u> &inside_programs,
  const std::array<ID3D11ComputeShader *, 3u> &roi_programs) {
  require(body_witness_research_configured());require(all_body_bits && raw_m && code1_m);
  const UINT pixels = width * height;
  std::map<std::string, std::vector<char>> original;
  constexpr std::array<const char *, 11u> names {"CurrentRGB.rgba16f", "OCR.u32", "SelectedBits.u32", "CoverBits.u32", "CBC.u32", "F2.u32", "FamilyModels.u32", "SCT.u32", "InsideBits.u32", "BODY.u32", "FinalM.u32"};
  for (UINT i = 0u; i < body_inputs.size(); ++i) {require(body_inputs[i]);original.emplace(names[i], read(body_inputs[i]));}
  original.emplace("AllBodyBits.u32", read(all_body_bits));original.emplace("RawM.u32", read(raw_m));original.emplace("Code1M.u32", read(code1_m));original.emplace("Constants.u32", read(constants));
  for (auto *shader : inside_programs) require(shader != nullptr);for (auto *shader : roi_programs) require(shader != nullptr);
  const std::array classifier_specs {
    cache::shader_spec {"tools/sbsbench/shaders/host_sbs_inside_topology_research_cs.hlsl", "prepare_main", "cs_5_0"},
    cache::shader_spec {"tools/sbsbench/shaders/host_sbs_inside_topology_research_cs.hlsl", "reduce_invalid_main", "cs_5_0"},
    cache::shader_spec {"tools/sbsbench/shaders/host_sbs_inside_topology_research_cs.hlsl", "classify_main", "cs_5_0"}};
  const auto snapshot = cache::snapshot_sources(body_witness_research_root(), classifier_specs);
  const auto closure = cache::source_closure_sha256(snapshot);std::array<ComPtr<ID3D11ComputeShader>, 3u> classifier;
  for (UINT i = 0u; i < classifier.size(); ++i) {const auto code = cache::get(snapshot, classifier_specs[i]);require(code && !code->empty());require(SUCCEEDED(device_->CreateComputeShader(code->data(), code->size(), nullptr, &classifier[i])));}
  inside_topology_probe_t result;
  result.closed = texture(width, height, DXGI_FORMAT_R32_UINT);result.exterior = texture(width, height, DXGI_FORMAT_R32_UINT);
  result.seeded = texture(width, height, DXGI_FORMAT_R32_UINT);result.unknown = texture(width, height, DXGI_FORMAT_R32_UINT);
  result.replay_inside = texture(width, height, DXGI_FORMAT_R32_UINT);result.header = topology_words(8u);
  auto eligible = texture(width, height, DXGI_FORMAT_R32_UINT), seeds = texture(width, height, DXGI_FORMAT_R32_UINT);
  auto domain = texture(width, height, DXGI_FORMAT_R32_UINT), cover = texture(width, height, DXGI_FORMAT_R32_UINT);
  auto graph_body = texture(width, height, DXGI_FORMAT_R32_UINT), graph_final = texture(width, height, DXGI_FORMAT_R32_UINT);
  auto mask_copy = texture(width, height, DXGI_FORMAT_R32_UINT);
  auto parent = topology_words(pixels + 1u), flags = topology_words(pixels + 1u), bad = topology_words(pixels + 1u);
  auto f2_copy = topology_words(68608u, true), arguments = topology_words(4u, true, true);
  auto shape = buffer(16u, true), hook = buffer(16u, true);
  const std::array<UINT, 4u> dims {width, height, pixels, 0u}, zero {};
  context_->UpdateSubresource(shape.resource.Get(), 0u, nullptr, dims.data(), 0u, 0u);
  for (auto *tex : {&result.closed, &result.exterior, &result.seeded, &result.unknown, &result.replay_inside,
         &eligible, &seeds, &domain, &cover, &graph_body, &graph_final}) {auto view = uav(tex->srv.Get());context_->ClearUnorderedAccessViewUint(view.Get(), zero.data());}
  context_->ClearUnorderedAccessViewUint(result.header.uav.Get(), zero.data());
  ComPtr<ID3D11Resource> source_f2, source_m;body_inputs[5u]->GetResource(&source_f2);body_inputs[10u]->GetResource(&source_m);
  context_->CopyResource(f2_copy.resource.Get(), source_f2.Get());context_->CopyResource(mask_copy.resource.Get(), source_m.Get());
  // Only private phase/counters reset. Source proofs, palettes and width data
  // remain copied verbatim and production F2 is never writable here.
  const std::array<UINT, 5u> phase_zero {};const D3D11_BOX phase {79u * 4u, 0u, 0u, 84u * 4u, 1u, 1u};
  context_->UpdateSubresource(f2_copy.resource.Get(), 0u, &phase, phase_zero.data(), 0u, 0u);
  const auto e_uav = uav(eligible.srv.Get()), s_uav = uav(seeds.srv.Get()), d_uav = uav(domain.srv.Get()), c_uav = uav(cover.srv.Get());
  const auto gb_uav = uav(graph_body.srv.Get()), gf_uav = uav(graph_final.srv.Get()), inside_uav = uav(result.replay_inside.srv.Get()), m_uav = uav(mask_copy.srv.Get());
  const auto closed_uav = uav(result.closed.srv.Get()), exterior_uav = uav(result.exterior.srv.Get()), seeded_uav = uav(result.seeded.srv.Get()), unknown_uav = uav(result.unknown.srv.Get());
  auto unbind = [&]() {std::array<ID3D11ShaderResourceView *, 16u> sr {};std::array<ID3D11UnorderedAccessView *, 8u> ur {};context_->CSSetUnorderedAccessViews(0u, UINT(ur.size()), ur.data(), nullptr);context_->CSSetShaderResources(0u, UINT(sr.size()), sr.data());};
  auto inside_stage = [&](UINT entry, bool direct) {
    std::array<ID3D11ShaderResourceView *, 15u> in {body_inputs[0u], all_body_bits, nullptr, body_inputs[3u], body_inputs[4u],
      eligible.srv.Get(), parent.srv.Get(), flags.srv.Get(), raw_m, code1_m, f2_copy.srv.Get(), nullptr, nullptr, body_inputs[6u], result.replay_inside.srv.Get()};
    std::array<ID3D11UnorderedAccessView *, 8u> out {};
    if (entry == 0u) out[7u] = arguments.uav.Get();
    if (entry == 1u) {out[0u] = e_uav.Get();out[1u] = s_uav.Get();out[2u] = d_uav.Get();out[3u] = c_uav.Get();out[4u] = parent.uav.Get();out[5u] = flags.uav.Get();in[5u] = in[6u] = in[7u] = nullptr;}
    if (entry == 2u || entry == 3u) {in[10u] = in[14u] = nullptr;out[6u] = inside_uav.Get();out[7u] = f2_copy.uav.Get();}
    if (entry == 3u) out[0u] = m_uav.Get();
    ID3D11Buffer *cb[3] {constants, hook.resource.Get(), shape.resource.Get()};
    context_->CSSetConstantBuffers(0u, 3u, cb);context_->CSSetShaderResources(0u, UINT(in.size()), in.data());context_->CSSetUnorderedAccessViews(0u, UINT(out.size()), out.data(), nullptr);
    context_->CSSetShader(inside_programs[entry], nullptr, 0u);
    if (direct) context_->Dispatch(entry == 3u ? (width + 15u) / 16u : 1u, entry == 3u ? (height + 15u) / 16u : 1u, 1u);
    else context_->DispatchIndirect(arguments.resource.Get(), 0u);unbind();
  };
  auto classify_stage = [&](UINT entry) {
    const std::array<ID3D11ShaderResourceView *, 7u> in {body_inputs[0u], eligible.srv.Get(), parent.srv.Get(), flags.srv.Get(), body_inputs[3u], body_inputs[4u], body_inputs[1u]};
    const std::array<ID3D11UnorderedAccessView *, 6u> out {closed_uav.Get(), exterior_uav.Get(), seeded_uav.Get(), unknown_uav.Get(), bad.uav.Get(), result.header.uav.Get()};
    ID3D11Buffer *cb[3] {constants, hook.resource.Get(), shape.resource.Get()};context_->CSSetConstantBuffers(0u, 3u, cb);
    context_->CSSetShaderResources(0u, UINT(in.size()), in.data());context_->CSSetUnorderedAccessViews(0u, UINT(out.size()), out.data(), nullptr);
    context_->CSSetShader(classifier[entry].Get(), nullptr, 0u);context_->DispatchIndirect(arguments.resource.Get(), 0u);unbind();
  };
  const bool export_graphs = std::getenv("SUNSHINE_TEST_INSIDE_TOPOLOGY_GRAPH_EXPORT") != nullptr;
  for (UINT unit = 0u; unit < 8u; ++unit) {
    const std::array<UINT, 4u> u {unit, 0u, 0u, 0u};context_->UpdateSubresource(hook.resource.Get(), 0u, nullptr, u.data(), 0u, 0u);
    context_->ClearUnorderedAccessViewUint(arguments.uav.Get(), zero.data());inside_stage(0u, true);inside_stage(1u, false);
    ID3D11Buffer *cb[3] {shape.resource.Get(), hook.resource.Get(), nullptr};
    const std::array<ID3D11ShaderResourceView *, 6u> in {eligible.srv.Get(), seeds.srv.Get(), domain.srv.Get(), cover.srv.Get(), body_inputs[4u], nullptr};
    const std::array<ID3D11UnorderedAccessView *, 4u> out {parent.uav.Get(), flags.uav.Get(), gb_uav.Get(), gf_uav.Get()};
    context_->CSSetConstantBuffers(0u, 3u, cb);context_->CSSetShaderResources(0u, UINT(in.size()), in.data());context_->CSSetUnorderedAccessViews(0u, UINT(out.size()), out.data(), nullptr);
    // Canonical Inside emit initializes Parent/Flags; only actual ROI union and
    // reduction follow, with explicit NULL physical-support t5 as production.
    for (const UINT entry : {1u, 2u}) {context_->CSSetShader(roi_programs[entry], nullptr, 0u);context_->DispatchIndirect(arguments.resource.Get(), 0u);}unbind();
    context_->ClearUnorderedAccessViewUint(bad.uav.Get(), zero.data());classify_stage(0u);classify_stage(1u);classify_stage(2u);
    if (export_graphs) {
      const auto issued = words(read(arguments.resource.Get()));
      if (issued.size() == 4u && issued[0u] != 0u && issued[1u] != 0u && issued[2u] != 0u) {
        const auto prefix = "unit" + std::to_string(unit) + "_";
        result.artifacts.emplace(prefix + "Parent.u32", read(parent.resource.Get()));result.artifacts.emplace(prefix + "RootFlags.u32", read(flags.resource.Get()));
        result.artifacts.emplace(prefix + "Eligible.u32", read(eligible.resource.Get()));result.artifacts.emplace(prefix + "RootBad.u32", read(bad.resource.Get()));
      }
    }
    inside_stage(2u, false);
  }
  inside_stage(3u, true);unbind();ID3D11Buffer *clear_cb[3] {};context_->CSSetConstantBuffers(0u, 3u, clear_cb);context_->CSSetShader(nullptr, nullptr, 0u);
  result.replay_words = words(read(result.replay_inside.resource.Get()));result.header_words = words(read(result.header.resource.Get()));
  result.byteexact = raw(result.replay_words) == original.at("InsideBits.u32");
  result.artifacts.emplace("replayed_InsideBits.u32", raw(result.replay_words));result.artifacts.emplace("UnitHeader.u32", raw(result.header_words));
  result.artifacts.emplace("closed_unseeded.u32", read(result.closed.resource.Get()));result.artifacts.emplace("exterior.u32", read(result.exterior.resource.Get()));
  result.artifacts.emplace("seeded.u32", read(result.seeded.resource.Get()));result.artifacts.emplace("unknown.u32", read(result.unknown.resource.Get()));
  result.artifacts.emplace("private_phase_F2.u32", read(f2_copy.resource.Get()));
  for (UINT i = 0u; i < body_inputs.size(); ++i) require(read(body_inputs[i]) == original.at(names[i]));
  require(read(all_body_bits) == original.at("AllBodyBits.u32"));require(read(raw_m) == original.at("RawM.u32"));require(read(code1_m) == original.at("Code1M.u32"));require(read(constants) == original.at("Constants.u32"));
  nlohmann::json bindings = nlohmann::json::object();for (const auto &[name, bytes] : original) {bindings[name] = {{"bytes", bytes.size()}, {"sha256", hash_bytes(bytes)}};result.artifacts.emplace("consumed_" + name, bytes);}
  result.metadata = {{"schema", 1u}, {"classifier_source_closure", closure}, {"inside_bits_byteexact_to_production", result.byteexact},
    {"production_inputs_byteexact_after_replay", true}, {"consumed_input_bindings", bindings},
    {"scope", "PRIVATE canonical Inside prepare/emit/publish/finalize plus the exact canonical ROI union/reduce shaders in fresh scratch; all eight units; actual Parent/RootFlags classifier"},
    {"canonical_inside_program_source_closure", inside_closure_}, {"canonical_roi_program_source_closure", models::current_outline_runtime::roi_closure},
    {"topology_interpretation_permitted", result.byteexact}, {"unit_header_words", result.header_words},
    {"unit_header_status", "0 unissued;1 prepared;any bit2 revokes all labels for that unit"},
    {"invalid_source_boundary_policy", "Any invalid immediate native N4 source neighbor of an actual eligible member marks its whole actual root unknown before ordered classification"},
    {"alpha_or_render_authority", false}, {"quality_accepted", false}};
  RecordProperty("inside_topology_replay_byteexact", result.byteexact ? "true" : "false");return result;
}

body_witness_probe_t probe_body_with_topology(ID3D11Buffer *constants, UINT width, UINT height,
  const std::array<ID3D11ShaderResourceView *, 11u> &inputs, const inside_topology_probe_t &topology, bool physical_k = false) {
  require(body_witness_research_configured());
  constexpr std::array<const char *, 11u> names {"CurrentRGB.rgba16f", "OCR.u32", "SelectedBits.u32", "CoverBits.u32", "CBC.u32", "F2.u32", "FamilyModels.u32", "SCT.u32", "InsideBits.u32", "BODY.u32", "FinalM.u32"};
  const std::array specs {physical_k ? cache::shader_spec {"tools/sbsbench/shaders/host_sbs_unanchored_body_physical_k_research_cs.hlsl", "unanchored_body_physical_k_main", "cs_5_0"} :
    cache::shader_spec {"tools/sbsbench/shaders/host_sbs_unanchored_body_topology_research_cs.hlsl", "unanchored_body_topology_main", "cs_5_0"}};
  const auto snapshot = cache::snapshot_sources(body_witness_research_root(), specs);const auto closure = cache::source_closure_sha256(snapshot);
  if (const auto *pin = std::getenv(physical_k ? "SUNSHINE_TEST_PHYSICAL_K_TOPOLOGY_FILTER_PIN" : "SUNSHINE_TEST_INSIDE_TOPOLOGY_FILTER_PIN")) require(closure == pin);
  std::array<std::vector<char>, 11u> before;for (UINT i = 0u; i < inputs.size(); ++i) {require(inputs[i]);before[i] = read(inputs[i]);}
  const auto constants_before = read(constants);const auto count = std::size_t(width) * height;
  // A replay's positive parity applies only to the exact operands it consumed.
  // Never let cached topology A authorize another valid source/context B, or
  // an overridden texture with the same C24 shape/frame annotations.
  bool source_matches = true;nlohmann::json mismatches = nlohmann::json::array();
  for (UINT i = 0u; i < inputs.size(); ++i) {
    const auto found = topology.artifacts.find("consumed_" + std::string(names[i]));
    if (found == topology.artifacts.end() || found->second != before[i]) {source_matches = false;mismatches.push_back(names[i]);}
  }
  const auto expected_constants = topology.artifacts.find("consumed_Constants.u32");
  if (expected_constants == topology.artifacts.end() || expected_constants->second != constants_before) {source_matches = false;mismatches.push_back("Constants.u32");}
  const bool usable = topology.byteexact && source_matches;
  body_witness_probe_t result;result.rejected.assign(count, 0u);result.filtered = words(before[10u]);result.reasons.assign(count, 2u | 1024u);
  const std::array<UINT, 4u> gate {0x31504f54u, 1u, usable ? 1u : 0u, 0u};
  if (usable) {
    const auto code = cache::get(snapshot, specs[0u]);require(code && !code->empty());ComPtr<ID3D11ComputeShader> shader;
    require(SUCCEEDED(device_->CreateComputeShader(code->data(), code->size(), nullptr, &shader)));
    auto rejected = texture(width, height, DXGI_FORMAT_R32_UINT), filtered = texture(width, height, DXGI_FORMAT_R32_UINT), reasons = texture(width, height, DXGI_FORMAT_R32_UINT);
    auto ru = uav(rejected.srv.Get()), fu = uav(filtered.srv.Get()), ku = uav(reasons.srv.Get());
    auto shape = buffer(16u, true), permission = buffer(16u, true);const std::array<UINT, 4u> dims {width, height, width * height, 0u};
    context_->UpdateSubresource(shape.resource.Get(), 0u, nullptr, dims.data(), 0u, 0u);context_->UpdateSubresource(permission.resource.Get(), 0u, nullptr, gate.data(), 0u, 0u);
    std::array<ID3D11ShaderResourceView *, 17u> in {};std::copy_n(inputs.begin(), 8u, in.begin());std::copy_n(inputs.begin() + 8u, 3u, in.begin() + 9u);
    in[12u] = topology.closed.srv.Get();in[13u] = topology.unknown.srv.Get();in[14u] = topology.header.srv.Get();in[15u] = topology.exterior.srv.Get();in[16u] = topology.seeded.srv.Get();
    std::array<ID3D11UnorderedAccessView *, 5u> out {nullptr, nullptr, ru.Get(), fu.Get(), ku.Get()};ID3D11Buffer *cb[3] {constants, shape.resource.Get(), permission.resource.Get()};
    context_->CSSetConstantBuffers(0u, 3u, cb);context_->CSSetShaderResources(0u, UINT(in.size()), in.data());context_->CSSetUnorderedAccessViews(0u, UINT(out.size()), out.data(), nullptr);
    context_->CSSetShader(shader.Get(), nullptr, 0u);context_->Dispatch((width + 15u) / 16u, (height + 15u) / 16u, 1u);
    out.fill(nullptr);in.fill(nullptr);context_->CSSetUnorderedAccessViews(0u, UINT(out.size()), out.data(), nullptr);context_->CSSetShaderResources(0u, UINT(in.size()), in.data());
    ID3D11Buffer *clear[3] {};context_->CSSetConstantBuffers(0u, 3u, clear);context_->CSSetShader(nullptr, nullptr, 0u);
    result.rejected = words(read(rejected.resource.Get()));result.filtered = words(read(filtered.resource.Get()));result.reasons = words(read(reasons.resource.Get()));
  }
  require(result.filtered.size() == count && result.rejected.size() == count && result.reasons.size() == count);
  nlohmann::json bindings = nlohmann::json::object();
  for (UINT i = 0u; i < inputs.size(); ++i) {require(read(inputs[i]) == before[i]);bindings[names[i]] = {{"bytes", before[i].size()}, {"sha256", hash_bytes(before[i])}};result.consumed.emplace(names[i], std::move(before[i]));}
  require(read(constants) == constants_before);result.consumed.emplace("Constants.u32", constants_before);result.consumed.emplace("TopologyGate.u32", raw(gate));
  bindings["Constants.u32"] = {{"bytes", constants_before.size()}, {"sha256", hash_bytes(constants_before)}};
  result.consumed.emplace("TopologyClosed.u32", read(topology.closed.resource.Get()));result.consumed.emplace("TopologyUnknown.u32", read(topology.unknown.resource.Get()));
  result.consumed.emplace("TopologyHeader.u32", read(topology.header.resource.Get()));result.consumed.emplace("TopologyExterior.u32", read(topology.exterior.resource.Get()));result.consumed.emplace("TopologySeeded.u32", read(topology.seeded.resource.Get()));
  const auto original = words(result.consumed.at("FinalM.u32"));UINT proposals = 0u, withdrawals = 0u, additions = 0u;
  for (std::size_t i = 0u; i < count; ++i) {proposals += result.rejected[i] != 0u;withdrawals += original[i] == 1u && result.filtered[i] == 0u;additions += original[i] == 0u && result.filtered[i] == 1u;}
  result.metadata = {{"schema", 1u}, {"source_closure", closure}, {"source_tuple_byteexact_after_probe", true}, {"shape", {width, height}},
    {"scope", physical_k ? "PRIVATE body-witness proposal restricted to separate physical-K CLOSED-UNSEEDED graph after exact canonical Inside control; exterior/qualified-FG/current-Inside/invalid-neighbor/unknown retained" : "PRIVATE body-witness proposal restricted to actual canonical Inside CLOSED-UNSEEDED components; original/exterior/seeded/invalid-neighbor/unknown evidence retained"},
    {"topology_replay", topology.metadata}, {"topology_consumed_inputs_match", source_matches}, {"topology_consumed_input_mismatches", mismatches},
    {"topology_current_tuple_usable", usable}, {"filter_dispatched", usable}, {"production_inputs_byteexact_after_probe", true},
    {"consumed_input_bindings", bindings}, {"mask_delta", {{"proposed_rejected_body_pixels", proposals}, {"actual_M_withdrawals", withdrawals}, {"actual_M_additions", additions}}},
    {"reason_bits", {{"body", 1u}, {"keep_unknown", 2u}, {"keep_exact_foreground", 4u}, {"keep_existing_inside", 8u}, {"keep_blacklike", 16u}, {"keep_other_unit", 32u},
      {"reject", 64u}, {"original_high_present", 128u}, {"other_covered_unit_present", 256u}, {"keep_not_closed_unseeded", 512u}, {"keep_topology_unknown", 1024u}, {"keep_topology_exterior", 2048u}, {"keep_topology_seeded", 4096u}, {"certified_closed_unseeded", 8192u}}},
    {"candidate_is_production_mask", false}, {"current_production_M_is_alpha_truth", false}, {"quality_accepted", false}};
  return result;
}

std::array<ID3D11ShaderResourceView *, 11u> current_body_witness_inputs(ID3D11ShaderResourceView *rgb_override = nullptr) {
  const auto e = helper_.enclosure_views();const auto f = helper_.mutable_views();
  return {rgb_override ? rgb_override : rgb_.srv.Get(), ocr_.srv.Get(), e.selected_bits.Get(), e.cover_bits.Get(), e.receipt.Get(),
    f.calibration_data.Get(), e.family_models.Get(), sct_.srv.Get(), f.inside_bits.Get(), e.body.Get(), f.matte.Get()};
}

std::pair<inside_topology_probe_t, body_witness_probe_t> probe_current_inside_topology(ID3D11ShaderResourceView *rgb_override = nullptr) {
  const auto e = helper_.enclosure_views();const auto r = helper_.raw_views();const auto inputs = current_body_witness_inputs(rgb_override);
  auto topology = replay_inside_topology(cb_.resource.Get(), W_, H_, inputs, e.all_body_bits.Get(), r.matte.Get(), e.matte.Get(), program_.inside_shaders, program_.roi_graph);
  auto filtered = probe_body_with_topology(cb_.resource.Get(), W_, H_, inputs, topology);return {std::move(topology), std::move(filtered)};
}

void export_inside_topology_probe(const char *label, const inside_topology_probe_t &topology, const body_witness_probe_t &filtered, const char *scope, bool physical_k = false) {
  nlohmann::json report = filtered.metadata;report["fixture_scope"] = scope;
  const auto *root = std::getenv(physical_k ? "SUNSHINE_TEST_PHYSICAL_K_TOPOLOGY_OUTPUT" : "SUNSHINE_TEST_INSIDE_TOPOLOGY_OUTPUT");
  if (root && *root) {
    const auto folder = std::filesystem::path(root) / label;require(!std::filesystem::exists(folder));require(std::filesystem::create_directories(folder));
    nlohmann::json artifacts = nlohmann::json::object();
    const auto emit = [&](const std::string &name, const std::vector<char> &bytes) {
      const auto path = folder / name;std::ofstream stream(path, std::ios::binary);stream.write(bytes.data(), std::streamsize(bytes.size()));stream.close();require(bool(stream));
      artifacts[name] = {{"path", path.string()}, {"bytes", bytes.size()}, {"sha256", hash_bytes(bytes)}};
    };
    for (const auto &[name, bytes] : topology.artifacts) emit("topology_" + name, bytes);
    for (const auto &[name, bytes] : filtered.consumed) emit("filter_consumed_" + name, bytes);
    emit("filtered_M.u32", raw(filtered.filtered));emit("rejected.u32", raw(filtered.rejected));emit("reasons.u32", raw(filtered.reasons));
    report["artifacts"] = artifacts;std::ofstream stream(folder / "report.json", std::ios::binary);stream << report.dump(2) << '\n';stream.close();require(bool(stream));
  }
  RecordProperty(std::string(label) + (physical_k ? "_physical_k_topology" : "_inside_topology"), report.dump());
}

UINT author_topology_scene_counter(UINT frame, bool open) {
  author_band(5u, frame);
  // Separate authored SCENE object below the caption's colour-learning core.
  // It is not a glyph or a generated detector/mask input. The actual helper
  // decides whether its ordinary colour candidate reaches current BODY.
  rect(66u, 98u, 82u, 110u, {0u, 0u, 0u, 255u});
  rect(70u, 101u, 78u, 107u, {160u, 160u, 160u, 255u});
  if (open) rect(73u, 106u, 75u, 113u, {255u, 160u, 255u, 255u});
  upload_pixels_from_origin();return 103u * W_ + 74u;
}
