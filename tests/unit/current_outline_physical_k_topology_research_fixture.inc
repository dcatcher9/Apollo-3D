// PRIVATE source-identical physical-K graph. The original canonical Inside
// replay remains an independent mandatory control; only fresh scratch is writable.
inside_topology_probe_t replay_physical_k_topology(ID3D11Buffer *constants, UINT width, UINT height,
  const std::array<ID3D11ShaderResourceView *, 11u> &inputs, const inside_topology_probe_t &canonical,
  const std::array<ID3D11ComputeShader *, 3u> &roi_programs) {
  require(body_witness_research_configured());const UINT pixels = width * height;
  constexpr std::array<const char *, 11u> names {"CurrentRGB.rgba16f", "OCR.u32", "SelectedBits.u32", "CoverBits.u32", "CBC.u32", "F2.u32", "FamilyModels.u32", "SCT.u32", "InsideBits.u32", "BODY.u32", "FinalM.u32"};
  std::map<std::string, std::vector<char>> original;
  for (UINT i = 0u; i < inputs.size(); ++i) {require(inputs[i]);original.emplace(names[i], read(inputs[i]));}
  original.emplace("Constants.u32", read(constants));bool same_source = true;nlohmann::json mismatches = nlohmann::json::array();
  for (const auto &[name, bytes] : original) {
    const auto found = canonical.artifacts.find("consumed_" + name);
    if (found == canonical.artifacts.end() || found->second != bytes) {same_source = false;mismatches.push_back(name);}
  }
  const bool usable = canonical.byteexact && same_source;
  const std::array producer_specs {
    cache::shader_spec {"tools/sbsbench/shaders/host_sbs_physical_k_topology_research_cs.hlsl", "physical_prepare_main", "cs_5_0"},
    cache::shader_spec {"tools/sbsbench/shaders/host_sbs_physical_k_topology_research_cs.hlsl", "physical_emit_main", "cs_5_0"}};
  const std::array classifier_specs {
    cache::shader_spec {"tools/sbsbench/shaders/host_sbs_inside_topology_research_cs.hlsl", "prepare_main", "cs_5_0"},
    cache::shader_spec {"tools/sbsbench/shaders/host_sbs_inside_topology_research_cs.hlsl", "reduce_invalid_main", "cs_5_0"},
    cache::shader_spec {"tools/sbsbench/shaders/host_sbs_inside_topology_research_cs.hlsl", "classify_main", "cs_5_0"}};
  const auto producer_snapshot = cache::snapshot_sources(body_witness_research_root(), producer_specs);
  const auto classifier_snapshot = cache::snapshot_sources(body_witness_research_root(), classifier_specs);
  const auto graph_closure = cache::source_closure_sha256(producer_snapshot), classifier_closure = cache::source_closure_sha256(classifier_snapshot);
  if (const auto *pin = std::getenv("SUNSHINE_TEST_PHYSICAL_K_TOPOLOGY_GRAPH_PIN")) require(graph_closure == pin);
  if (const auto *pin = std::getenv("SUNSHINE_TEST_INSIDE_TOPOLOGY_CLASSIFIER_PIN")) require(classifier_closure == pin);
  std::array<ComPtr<ID3D11ComputeShader>, 2u> producer;std::array<ComPtr<ID3D11ComputeShader>, 3u> classifier;
  if (usable) {
    for (UINT i = 0u; i < producer.size(); ++i) {const auto code = cache::get(producer_snapshot, producer_specs[i]);require(code && !code->empty());require(SUCCEEDED(device_->CreateComputeShader(code->data(), code->size(), nullptr, &producer[i])));}
    for (UINT i = 0u; i < classifier.size(); ++i) {const auto code = cache::get(classifier_snapshot, classifier_specs[i]);require(code && !code->empty());require(SUCCEEDED(device_->CreateComputeShader(code->data(), code->size(), nullptr, &classifier[i])));}
  }
  inside_topology_probe_t result;result.byteexact = usable;
  result.closed = texture(width, height, DXGI_FORMAT_R32_UINT);result.exterior = texture(width, height, DXGI_FORMAT_R32_UINT);
  result.seeded = texture(width, height, DXGI_FORMAT_R32_UINT);result.unknown = texture(width, height, DXGI_FORMAT_R32_UINT);result.header = topology_words(8u);
  auto eligible = texture(width, height, DXGI_FORMAT_R32_UINT), seeds = texture(width, height, DXGI_FORMAT_R32_UINT);
  auto domain = texture(width, height, DXGI_FORMAT_R32_UINT), cover = texture(width, height, DXGI_FORMAT_R32_UINT);
  auto graph_body = texture(width, height, DXGI_FORMAT_R32_UINT), graph_final = texture(width, height, DXGI_FORMAT_R32_UINT);
  auto parent = topology_words(pixels + 1u), flags = topology_words(pixels + 1u), bad = topology_words(pixels + 1u), arguments = topology_words(4u, true, true);
  auto shape = buffer(16u, true), hook = buffer(16u, true);const std::array<UINT, 4u> dims {width, height, pixels, 0u}, zero {};
  context_->UpdateSubresource(shape.resource.Get(), 0u, nullptr, dims.data(), 0u, 0u);
  for (auto *tex : {&result.closed, &result.exterior, &result.seeded, &result.unknown, &eligible, &seeds, &domain, &cover, &graph_body, &graph_final}) {
    auto view = uav(tex->srv.Get());context_->ClearUnorderedAccessViewUint(view.Get(), zero.data());
  }
  context_->ClearUnorderedAccessViewUint(result.header.uav.Get(), zero.data());
  const auto eu = uav(eligible.srv.Get()), su = uav(seeds.srv.Get()), du = uav(domain.srv.Get()), cu = uav(cover.srv.Get());
  const auto gbu = uav(graph_body.srv.Get()), gfu = uav(graph_final.srv.Get());
  const auto closed_u = uav(result.closed.srv.Get()), exterior_u = uav(result.exterior.srv.Get()), seeded_u = uav(result.seeded.srv.Get()), unknown_u = uav(result.unknown.srv.Get());
  auto unbind = [&]() {std::array<ID3D11ShaderResourceView *, 16u> in {};std::array<ID3D11UnorderedAccessView *, 8u> out {};context_->CSSetUnorderedAccessViews(0u, UINT(out.size()), out.data(), nullptr);context_->CSSetShaderResources(0u, UINT(in.size()), in.data());};
  auto produce = [&](UINT entry) {
    // t1 satisfies canonical shape validation only; the private seed predicate
    // never reads it. Exact qualified FG/current Inside replace BODY seeds.
    std::array<ID3D11ShaderResourceView *, 16u> in {};in[0u] = inputs[0u];in[1u] = inputs[2u];in[3u] = inputs[3u];in[4u] = inputs[4u];
    in[10u] = inputs[5u];in[13u] = inputs[6u];in[15u] = inputs[8u];
    std::array<ID3D11UnorderedAccessView *, 8u> out {};out[4u] = parent.uav.Get();out[5u] = flags.uav.Get();
    if (entry == 0u) out[7u] = arguments.uav.Get();
    else {out[0u] = eu.Get();out[1u] = su.Get();out[2u] = du.Get();out[3u] = cu.Get();}
    ID3D11Buffer *cb[3] {constants, hook.resource.Get(), shape.resource.Get()};context_->CSSetConstantBuffers(0u, 3u, cb);
    context_->CSSetShaderResources(0u, UINT(in.size()), in.data());context_->CSSetUnorderedAccessViews(0u, UINT(out.size()), out.data(), nullptr);context_->CSSetShader(producer[entry].Get(), nullptr, 0u);
    if (entry == 0u) context_->Dispatch(1u, 1u, 1u);else context_->DispatchIndirect(arguments.resource.Get(), 0u);unbind();
  };
  auto classify = [&](UINT entry) {
    const std::array<ID3D11ShaderResourceView *, 7u> in {inputs[0u], eligible.srv.Get(), parent.srv.Get(), flags.srv.Get(), inputs[3u], inputs[4u], inputs[1u]};
    const std::array<ID3D11UnorderedAccessView *, 6u> out {closed_u.Get(), exterior_u.Get(), seeded_u.Get(), unknown_u.Get(), bad.uav.Get(), result.header.uav.Get()};
    ID3D11Buffer *cb[3] {constants, hook.resource.Get(), shape.resource.Get()};context_->CSSetConstantBuffers(0u, 3u, cb);
    context_->CSSetShaderResources(0u, UINT(in.size()), in.data());context_->CSSetUnorderedAccessViews(0u, UINT(out.size()), out.data(), nullptr);
    context_->CSSetShader(classifier[entry].Get(), nullptr, 0u);context_->DispatchIndirect(arguments.resource.Get(), 0u);unbind();
  };
  const bool export_graphs = std::getenv("SUNSHINE_TEST_PHYSICAL_K_TOPOLOGY_GRAPH_EXPORT") != nullptr;
  if (usable) for (UINT unit = 0u; unit < 8u; ++unit) {
    const std::array<UINT, 4u> u {unit, 0u, 0u, 0u};context_->UpdateSubresource(hook.resource.Get(), 0u, nullptr, u.data(), 0u, 0u);
    context_->ClearUnorderedAccessViewUint(arguments.uav.Get(), zero.data());produce(0u);produce(1u);
    ID3D11Buffer *cb[3] {shape.resource.Get(), hook.resource.Get(), nullptr};const std::array<ID3D11ShaderResourceView *, 6u> in {eligible.srv.Get(), seeds.srv.Get(), domain.srv.Get(), cover.srv.Get(), inputs[4u], nullptr};
    const std::array<ID3D11UnorderedAccessView *, 4u> out {parent.uav.Get(), flags.uav.Get(), gbu.Get(), gfu.Get()};
    context_->CSSetConstantBuffers(0u, 3u, cb);context_->CSSetShaderResources(0u, UINT(in.size()), in.data());context_->CSSetUnorderedAccessViews(0u, UINT(out.size()), out.data(), nullptr);
    // Identical actual native ROI graph, explicit NULL physical-support alias.
    for (const UINT entry : {1u, 2u}) {require(roi_programs[entry]);context_->CSSetShader(roi_programs[entry], nullptr, 0u);context_->DispatchIndirect(arguments.resource.Get(), 0u);}unbind();
    context_->ClearUnorderedAccessViewUint(bad.uav.Get(), zero.data());classify(0u);classify(1u);classify(2u);
    if (export_graphs) {
      const auto issued = words(read(arguments.resource.Get()));
      if (issued.size() == 4u && issued[0u] && issued[1u] && issued[2u]) {
        const auto prefix = "unit" + std::to_string(unit) + "_";result.artifacts.emplace(prefix + "Args.u32", raw(issued));
        result.artifacts.emplace(prefix + "Parent.u32", read(parent.resource.Get()));result.artifacts.emplace(prefix + "RootFlags.u32", read(flags.resource.Get()));
        result.artifacts.emplace(prefix + "Eligible.u32", read(eligible.resource.Get()));result.artifacts.emplace(prefix + "Seeds.u32", read(seeds.resource.Get()));result.artifacts.emplace(prefix + "RootBad.u32", read(bad.resource.Get()));
      }
    }
  }
  unbind();ID3D11Buffer *clear[3] {};context_->CSSetConstantBuffers(0u, 3u, clear);context_->CSSetShader(nullptr, nullptr, 0u);
  result.header_words = words(read(result.header.resource.Get()));result.artifacts.emplace("UnitHeader.u32", raw(result.header_words));
  result.artifacts.emplace("closed_unseeded.u32", read(result.closed.resource.Get()));result.artifacts.emplace("exterior.u32", read(result.exterior.resource.Get()));
  result.artifacts.emplace("seeded.u32", read(result.seeded.resource.Get()));result.artifacts.emplace("unknown.u32", read(result.unknown.resource.Get()));
  // Retain the rejected broad-topology control independently, including all its
  // exact inputs and canonical replay output. It is never relabelled physical K.
  for (const auto &[name, bytes] : canonical.artifacts) result.artifacts.emplace("canonical_control_" + name, bytes);
  nlohmann::json bindings = nlohmann::json::object();
  for (UINT i = 0u; i < inputs.size(); ++i) require(read(inputs[i]) == original.at(names[i]));require(read(constants) == original.at("Constants.u32"));
  for (const auto &[name, bytes] : original) {bindings[name] = {{"bytes", bytes.size()}, {"sha256", hash_bytes(bytes)}};result.artifacts.emplace("consumed_" + name, bytes);}
  result.metadata = {{"schema", 1u}, {"graph_source_closure", graph_closure}, {"classifier_source_closure", classifier_closure}, {"canonical_inside_control", canonical.metadata},
    {"inside_bits_byteexact_to_production", canonical.byteexact}, {"topology_interpretation_permitted", usable}, {"production_inputs_byteexact_after_replay", true},
    {"canonical_control_consumed_inputs_match", same_source}, {"canonical_control_consumed_input_mismatches", mismatches}, {"consumed_input_bindings", bindings}, {"unit_header_words", result.header_words},
    {"canonical_roi_program_source_closure", models::current_outline_runtime::roi_closure},
    {"scope", "PRIVATE non-K valid-source graph; exact canonical PhysicalBlackRGB barriers, qualified exact FG or current fault-free Inside seeds; actual canonical native ROI union/reduce; existing invalid-neighbor/root-fault classifier"},
    {"invalid_source_boundary_policy", "Any invalid immediate native N4 neighbour marks the entire actual root unknown; ROI/Cover/native boundary remains exterior"},
    {"semantic_ambiguity", "Unseeded genuine glyphs/accents or halo regions enclosed by physical K may remain closed; no semantic alpha certainty is claimed"}, {"alpha_or_render_authority", false}, {"quality_accepted", false}};
  return result;
}

std::pair<inside_topology_probe_t, body_witness_probe_t> probe_current_physical_k_topology(ID3D11ShaderResourceView *rgb_override = nullptr) {
  const auto e = helper_.enclosure_views();const auto r = helper_.raw_views();const auto inputs = current_body_witness_inputs(rgb_override);
  const auto canonical = replay_inside_topology(cb_.resource.Get(), W_, H_, inputs, e.all_body_bits.Get(), r.matte.Get(), e.matte.Get(), program_.inside_shaders, program_.roi_graph);
  auto topology = replay_physical_k_topology(cb_.resource.Get(), W_, H_, inputs, canonical, program_.roi_graph);
  auto filtered = probe_body_with_topology(cb_.resource.Get(), W_, H_, inputs, topology, true);return {std::move(topology), std::move(filtered)};
}

void attach_physical_k_authored_alpha(inside_topology_probe_t &topology, body_witness_probe_t &filtered, bool detached_glyph) {
  // Independent input-layer rasterization for author_band(5) plus the explicitly
  // authored scene/glyph ring. No detector, BODY, graph or M values determine alpha.
  std::vector<std::uint8_t> alpha(std::size_t(W_) * H_, 0u);
  std::vector<pixel8> clean(alpha.size(), {128u, 128u, 128u, 255u});
  const auto paint = [&](UINT l, UINT t, UINT r, UINT b) {for (UINT y = t; y < b; ++y) for (UINT x = l; x < r; ++x) alpha[y * W_ + x] = 255u;};
  paint(74u, 43u, 116u, 93u);paint(155u, 43u, 213u, 93u);
  if (detached_glyph) paint(66u, 98u, 82u, 110u);
  else for (UINT y = 98u; y < 110u; ++y) for (UINT x = 66u; x < 82u; ++x) clean[y * W_ + x] = origin_pixels_[y * W_ + x];
  topology.artifacts.emplace("authored_alpha.u8", raw(alpha));topology.artifacts.emplace("authored_clean_scene.rgba8", raw(clean));topology.artifacts.emplace("authored_source.rgba8", raw(origin_pixels_));
  filtered.metadata["independent_authored_truth"] = {{"alpha_artifact", "topology_authored_alpha.u8"}, {"source_artifact", "topology_authored_source.rgba8"}, {"clean_scene_artifact", "topology_authored_clean_scene.rgba8"},
    {"authoring", "Opaque original black/white band glyph rectangles, plus independently declared detached outlined glyph or scene ring. True layers are never detector inputs."},
    {"detached_ring_role", detached_glyph ? "caption glyph alpha255" : "scene alpha0"}, {"target_xy", {74u, 103u}}, {"target_alpha", detached_glyph ? 255u : 0u}, {"target_source_clean_max_RGB8_delta", detached_glyph ? 32u : 0u}};
}
