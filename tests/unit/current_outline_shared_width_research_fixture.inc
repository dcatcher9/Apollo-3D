// Included inside the existing real-helper GPU fixture. Test-only observer;
// immutable producer views never become UAVs in this experiment.
struct shared_width_probe_t {
  std::vector<UINT> prior, candidate;
  nlohmann::json metadata;
};

bool shared_width_research_configured() const {
  const auto *root = std::getenv("SUNSHINE_TEST_SHARED_WIDTH_REPO_ROOT");
  return root && *root;
}

shared_width_probe_t probe_shared_width(
  ID3D11Buffer *constants, UINT width, UINT height,
  const std::array<ID3D11ShaderResourceView *, 8u> &inputs
) {
  require(shared_width_research_configured());
  const auto root = std::filesystem::path(std::getenv("SUNSHINE_TEST_SHARED_WIDTH_REPO_ROOT"));
  constexpr const char *file = "tools/sbsbench/shaders/host_sbs_shared_width_prior_research_cs.hlsl";
  const std::array specs {cache::shader_spec {file, "prior_main", "cs_5_0"},
    cache::shader_spec {file, "candidate_main", "cs_5_0"}};
  const auto snapshot = cache::snapshot_sources(root, specs);
  const auto closure = cache::source_closure_sha256(snapshot);
  if (const auto *expected = std::getenv("SUNSHINE_TEST_SHARED_WIDTH_SHADER_PIN")) require(closure == expected);
  std::array<ComPtr<ID3D11ComputeShader>, 2u> shaders;
  for (UINT i = 0u; i < shaders.size(); ++i) {
    const auto code = cache::get(snapshot, specs[i]);require(code && !code->empty());
    require(SUCCEEDED(device_->CreateComputeShader(code->data(), code->size(), nullptr, &shaders[i])));
  }
  std::array<std::vector<char>, 8u> before;
  for (UINT i = 0u; i < inputs.size(); ++i) {require(inputs[i] != nullptr);before[i] = read(inputs[i]);}
  const auto cb_before = read(constants);
  auto prior = texture(8u, 16u, DXGI_FORMAT_R32_UINT), candidate = texture(width, height, DXGI_FORMAT_R32_UINT);
  auto prior_uav = uav(prior.srv.Get()), candidate_uav = uav(candidate.srv.Get());
  auto shape = buffer(16u, true);
  const std::array<UINT, 4u> dims {width, height, width * height, 0u};
  context_->UpdateSubresource(shape.resource.Get(), 0u, nullptr, dims.data(), 0u, 0u);
  const std::array<UINT, 4u> zero {};
  context_->ClearUnorderedAccessViewUint(prior_uav.Get(), zero.data());
  context_->ClearUnorderedAccessViewUint(candidate_uav.Get(), zero.data());
  ID3D11Buffer *cb[2] {constants, shape.resource.Get()};
  std::array<ID3D11ShaderResourceView *, 9u> in {};
  std::copy(inputs.begin(), inputs.end(), in.begin());
  std::array<ID3D11UnorderedAccessView *, 2u> out {prior_uav.Get(), nullptr};
  context_->CSSetConstantBuffers(0u, 2u, cb);
  context_->CSSetShaderResources(0u, UINT(in.size()), in.data());
  context_->CSSetUnorderedAccessViews(0u, UINT(out.size()), out.data(), nullptr);
  context_->CSSetShader(shaders[0u].Get(), nullptr, 0u);context_->Dispatch(1u, 1u, 1u);
  out.fill(nullptr);context_->CSSetUnorderedAccessViews(0u, UINT(out.size()), out.data(), nullptr);
  in[8u] = prior.srv.Get();out[1u] = candidate_uav.Get();
  context_->CSSetShaderResources(0u, UINT(in.size()), in.data());
  context_->CSSetUnorderedAccessViews(0u, UINT(out.size()), out.data(), nullptr);
  context_->CSSetShader(shaders[1u].Get(), nullptr, 0u);context_->Dispatch((width + 15u) / 16u, (height + 15u) / 16u, 1u);
  in.fill(nullptr);out.fill(nullptr);
  context_->CSSetUnorderedAccessViews(0u, UINT(out.size()), out.data(), nullptr);
  context_->CSSetShaderResources(0u, UINT(in.size()), in.data());
  ID3D11Buffer *clear[2] {};context_->CSSetConstantBuffers(0u, 2u, clear);context_->CSSetShader(nullptr, nullptr, 0u);
  shared_width_probe_t result;
  result.prior = words(read(prior.resource.Get()));result.candidate = words(read(candidate.resource.Get()));
  require(result.prior.size() == 128u && result.candidate.size() == std::size_t(width) * height);
  for (UINT i = 0u; i < inputs.size(); ++i) require(read(inputs[i]) == before[i]);
  require(read(constants) == cb_before);
  result.metadata = {{"schema", 1u}, {"source_closure", closure}, {"production_inputs_byteexact_after_probe", true},
    {"scope", "PRIVATE diagnostic current-style prior and original-K geometric candidate; no production routing or local KNOWN/F2 changes"},
    {"candidate_is_alpha_proof", false}, {"candidate_is_production_mask", false}, {"history_used", false},
    {"prior_shape", {8u, 16u}}, {"candidate_shape", {width, height}},
    {"prior_rows", {"width", "status", "donor_bits", "original_local_status", "native_core_height", "donor_band_count", "original_local_width", "rejection_reason_OR", "donor0_rejection", "donor1_rejection", "donor2_rejection", "donor3_rejection", "donor4_rejection", "donor5_rejection", "donor6_rejection", "donor7_rejection"}},
    {"donor_rejection_bits", {{"not_strong", 1u}, {"different_native_core_height", 2u}, {"different_K_box", 4u}, {"different_foreground_boxes", 8u}, {"self", 16u}, {"receiver_style_unavailable", 32u}}},
    {"status_values", {{"unavailable", 0u}, {"local", 1u}, {"shared", 2u}, {"conflict", 3u}, {"no_donor", 4u}, {"no_style", 5u}, {"no_fill", 6u}}}};
  RecordProperty("shared_width_research_source_closure", closure);
  return result;
}

shared_width_probe_t probe_shared_width_current() {
  const auto e = helper_.enclosure_views();const auto f = helper_.mutable_views();
  return probe_shared_width(cb_.resource.Get(), W_, H_, {rgb_.srv.Get(), ocr_.srv.Get(), e.selected_bits.Get(),
    e.cover_bits.Get(), e.receipt.Get(), f.calibration_data.Get(), e.family_models.Get(), sct_.srv.Get()});
}

shared_width_probe_t probe_shared_width_model_fixture(const std::vector<UINT> &models) {
  // Explicit kernel-level representation/corruption fixture. It never changes
  // a production resource or claims to be a measured current model.
  require(models.size() == 17u * 8u);auto family = texture(17u, 8u, DXGI_FORMAT_R32_UINT);
  context_->UpdateSubresource(family.resource.Get(), 0u, nullptr, models.data(), 17u * 4u, 0u);
  const auto e = helper_.enclosure_views();const auto f = helper_.mutable_views();
  return probe_shared_width(cb_.resource.Get(), W_, H_, {rgb_.srv.Get(), ocr_.srv.Get(), e.selected_bits.Get(),
    e.cover_bits.Get(), e.receipt.Get(), f.calibration_data.Get(), family.srv.Get(), sct_.srv.Get()});
}

struct shared_width_authored_t {
  std::string name;
  std::vector<pixel8> scene, ink;
  std::vector<std::uint8_t> alpha;
  UINT receiver = 1u;
  UINT receiver_left = 256u;
  bool source_identical_ambiguity = true;
};

shared_width_authored_t author_shared_width_case(const std::string &kind, bool reverse, UINT frame) {
  const bool conflict = kind == "mixed_widths";
  const UINT units = conflict ? 3u : 2u;
  author(true, frame, 0u, false, false, false, 768u, 128u, 3u);
  shared_width_authored_t truth;
  truth.name = kind + (reverse ? "_reverse_f" : "_forward_f") + std::to_string(frame);truth.receiver = conflict ? 2u : (reverse ? 0u : 1u);
  truth.receiver_left = conflict ? 512u : 256u;
  truth.scene.assign(std::size_t(W_) * H_, {128u, 128u, 128u, 255u});
  truth.ink.assign(truth.scene.size(), {0u, 0u, 0u, 255u});truth.alpha.assign(truth.scene.size(), 0u);
  const pixel8 black {0u, 0u, 0u, 255u}, white {255u, 255u, 255u, 255u};
  const auto paint = [&](UINT l, UINT t, UINT r, UINT b, pixel8 colour, UINT opacity) {
    require(l < r && t < b && r <= W_ && b <= H_);
    for (UINT y = t; y < b; ++y) for (UINT x = l; x < r; ++x) {
      truth.ink[y * W_ + x] = colour;truth.alpha[y * W_ + x] = std::uint8_t(opacity);
    }
  };
  for (UINT physical = 0u; physical < units; ++physical) {
    const UINT left = physical * 256u;const bool hard = physical + 1u == units;
    const UINT thickness = conflict && physical == 1u ? 7u : 5u;
    if (hard) for (UINT y = 16u; y < 112u; ++y) for (UINT x = left + 16u; x < left + 240u; ++x)
      truth.scene[y * W_ + x] = black;
    const pixel8 foreground = hard && kind == "mixed_style" ? pixel8 {255u, 255u, 0u, 255u} : white;
    if (!(hard && kind == "missing_fill")) {
      for (const UINT x : {48u, 160u}) {
        paint(left + x - thickness, 48u - thickness, left + x + 32u + thickness, 88u + thickness, black, 255u);
        paint(left + x, 48u, left + x + 32u, 88u, foreground, 255u);
      }
      if (hard && kind == "counter_scene") {
        // Authored transparent counter; it has the SAME source RGB as K.
        // The candidate's behavior is scored, never presented as semantic truth.
        paint(left + 58u, 59u, left + 70u, 75u, black, 0u);
      }
    }
  }
  origin_pixels_ = truth.scene;
  for (std::size_t i = 0u; i < origin_pixels_.size(); ++i) for (UINT channel = 0u; channel < 3u; ++channel) {
    const UINT a = truth.alpha[i];origin_pixels_[i][channel] = std::uint8_t((a * truth.ink[i][channel] + (255u - a) * truth.scene[i][channel] + 127u) / 255u);
  }
  // Finer authored analysis geometry; the production owner still creates all
  // actual palettes, LOW/HIGH, profiles, widths and masks from source pixels.
  cb_words_[4u] = 96u;cb_words_[5u] = 16u;cb_words_[6u] = 0u;cb_words_[7u] = 16u;
  cb_words_[8u] = cb_words_[9u] = 0u;cb_words_[10u] = 96u;cb_words_[11u] = 16u;
  ocr_words_[3u] = ocr_words_[4u] = units;ocr_words_[9u] = W_;ocr_words_[10u] = fullH_;
  std::copy_n(cb_words_.begin() + 4u, 4u, ocr_words_.begin() + 11u);
  for (UINT logical = 0u; logical < units; ++logical) {
    const UINT physical = reverse && !conflict ? units - 1u - logical : logical;
    const UINT left = physical * 32u;
    const std::array<UINT, 4u> core {left + 5u, 5u, left + 26u, 12u}, cover {left + 2u, 2u, left + 30u, 14u};
    std::copy(core.begin(), core.end(), ocr_words_.begin() + 16u + 8u * logical);
    std::copy(cover.begin(), cover.end(), ocr_words_.begin() + 144u + 8u * logical);
  }
  std::copy_n(cb_words_.begin(), 12u, sct_words_.begin() + 8u);
  std::copy_n(cb_words_.begin() + 4u, 4u, uip_words_.begin() + 12u);
  std::copy_n(cb_words_.begin() + 8u, 4u, uip_words_.begin() + 16u);uip_words_[3u] = (1u << units) - 1u;
  upload_pixels_from_origin();return truth;
}

nlohmann::json export_shared_width_case(const shared_width_authored_t &truth, const shared_width_probe_t &probe) {
  const auto data = calibration(), selected = words(read(helper_.enclosure_views().selected_bits.Get()));
  const auto literal_m = words(outline_matte());const UINT bit = 1u << truth.receiver;
  UINT candidate_ink = 0u, candidate_scene = 0u, new_ink = 0u, new_scene = 0u;
  UINT visible_ink = 0u, identical_ink = 0u, faint_ink = 0u;
  UINT new_visible_ink = 0u, new_identical_ink = 0u, new_faint_ink = 0u;
  for (std::size_t i = 0u; i < probe.candidate.size(); ++i) if ((probe.candidate[i] & bit) != 0u) {
    const bool ink = truth.alpha[i] != 0u;candidate_ink += ink;candidate_scene += !ink;
    const bool added = literal_m[i] == 0u;
    new_ink += ink && added;new_scene += !ink && added;
    UINT delta = 0u;
    for (UINT channel = 0u; channel < 3u; ++channel)
      delta = std::max(delta, UINT(std::abs(int(origin_pixels_[i][channel]) - int(truth.scene[i][channel]))));
    const bool visible = ink && truth.alpha[i] >= 5u && delta >= 3u;
    const bool identical = ink && delta == 0u;
    // Metric-only partition; neither alpha nor clean scene enters the shader.
    const bool faint = ink && !visible && !identical;
    visible_ink += visible;identical_ink += identical;faint_ink += faint;
    new_visible_ink += visible && added;new_identical_ink += identical && added;new_faint_ink += faint && added;
  }
  nlohmann::json result = probe.metadata;
  result["name"] = truth.name;result["receiver_unit"] = truth.receiver;
  result["authored_truth_scope"] = "Separately authored original scene/ink/alpha; black hard-region scene and original black outlines are source-identical";
  result["source_identical_ambiguity"] = truth.source_identical_ambiguity;
  result["candidate_truth_counts"] = {{"ink", candidate_ink}, {"scene", candidate_scene}, {"new_ink_vs_production", new_ink}, {"new_scene_vs_production", new_scene},
    {"ink_visible_alpha_ge5_source_delta_ge3", visible_ink}, {"ink_source_identical", identical_ink}, {"ink_faint_or_subthreshold", faint_ink},
    {"new_ink_visible_alpha_ge5_source_delta_ge3", new_visible_ink}, {"new_ink_source_identical", new_identical_ink}, {"new_ink_faint_or_subthreshold", new_faint_ink}};
  result["visibility_metric"] = "Independent authored alpha>=5 and max-channel encoded RGB8 source-minus-clean-scene magnitude>=3; source-identical means exact RGB equality. Diagnostic scoring only.";
  result["quality_delta_scope"] = "Candidate scene and newly selected scene versus original production M are separate counts; an already protected scene pixel is not a new false positive.";
  result["quality_accepted"] = false;
  result["quality_status"] = candidate_scene ? "REJECTED_AS_ALPHA_MASK_ATTACHED_SCENE_AMBIGUITY" : "MECHANICS_ONLY_NO_GENERAL_QUALITY_ACCEPTANCE";
  result["prior_words_row_major"] = probe.prior;
  const auto *root = std::getenv("SUNSHINE_TEST_SHARED_WIDTH_OUTPUT");
  if (root && *root) {
    const auto folder = std::filesystem::path(root) / truth.name;require(!std::filesystem::exists(folder));require(std::filesystem::create_directories(folder));
    nlohmann::json artifacts = nlohmann::json::object();
    const auto emit = [&](const char *name, const auto &values) {
      const auto bytes = raw(values);const auto path = folder / name;std::ofstream stream(path, std::ios::binary);
      stream.write(bytes.data(), std::streamsize(bytes.size()));stream.close();require(bool(stream));
      artifacts[name] = {{"path", path.string()}, {"bytes", bytes.size()}, {"sha256", hash_bytes(bytes)}};
    };
    emit("scene.rgba8", truth.scene);emit("ink.rgba8", truth.ink);emit("alpha.u8", truth.alpha);emit("source.rgba8", origin_pixels_);
    emit("prior.u32", probe.prior);emit("candidate.u32", probe.candidate);emit("original_selected.u32", selected);
    emit("production_M.u32", literal_m);emit("original_F2.u32", data);emit("C24.u32", cb_words_);emit("OCR.u32", ocr_words_);emit("SCT.u32", sct_words_);
    const auto e = helper_.enclosure_views();emit("CBC.u32", words(read(e.receipt.Get())));emit("Cover.u32", words(read(e.cover_bits.Get())));emit("FamilyModels.u32", words(read(e.family_models.Get())));
    result["artifacts"] = artifacts;std::ofstream stream(folder / "report.json", std::ios::binary);stream << result.dump(2) << '\n';stream.close();require(bool(stream));
  }
  RecordProperty(truth.name + "_shared_width", result.dump());return result;
}
