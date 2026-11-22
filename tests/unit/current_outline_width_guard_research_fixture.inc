// Private measured-width guard uses the existing physical graph/replay owner.
// Copied operand fixtures are explicitly kernel-only and never producer writes.
void annotate_width_guard(body_witness_probe_t &filtered) {
  filtered.metadata["experiment"] = "PRIVATE current-local-measured-width erasure guard";
  filtered.metadata["reason_bits"]["keep_local_width_unknown"] = 16384u;
  filtered.metadata["reason_bits"]["keep_measured_K_collar"] = 32768u;
  filtered.metadata["reason_bits"]["keep_width_ROI_or_source_unknown"] = 65536u;
  filtered.metadata["width_guard_scope"] = "Only revoke existing physical-graph erasure proposals; current local StrongDonorBands/CU+11 width; literal original M retained near exact current physical K or unknown evidence. No shared-width borrowing, generated outline or alpha authority.";
}

std::pair<inside_topology_probe_t, body_witness_probe_t> probe_width_guard_inputs(const std::array<ID3D11ShaderResourceView *, 11u> &inputs) {
  const auto e = helper_.enclosure_views();const auto r = helper_.raw_views();
  const auto canonical = replay_inside_topology(cb_.resource.Get(), W_, H_, inputs, e.all_body_bits.Get(), r.matte.Get(), e.matte.Get(), program_.inside_shaders, program_.roi_graph);
  auto topology = replay_physical_k_topology(cb_.resource.Get(), W_, H_, inputs, canonical, program_.roi_graph);
  auto filtered = probe_body_with_topology(cb_.resource.Get(), W_, H_, inputs, topology, true);annotate_width_guard(filtered);
  return {std::move(topology), std::move(filtered)};
}

void export_width_guard_probe(const char *label, inside_topology_probe_t &topology, body_witness_probe_t &filtered, const char *scope) {
  annotate_width_guard(filtered);export_inside_topology_probe(label, topology, filtered, scope, true);
}

void attach_width_guard_scene_truth(inside_topology_probe_t &topology, body_witness_probe_t &filtered, UINT qx, UINT qy) {
  // Caption geometry is authored first; everything else (including the test
  // scene object) is clean scene. These arrays never enter detector/graph inputs.
  std::vector<std::uint8_t> alpha(std::size_t(W_) * H_, 0u);auto clean = origin_pixels_;
  const auto glyph = [&](UINT l, UINT t, UINT r, UINT b) {for (UINT y = t; y < b; ++y) for (UINT x = l; x < r; ++x) {alpha[y * W_ + x] = 255u;clean[y * W_ + x] = {128u, 128u, 128u, 255u};}};
  glyph(74u, 43u, 116u, 93u);glyph(155u, 43u, 213u, 93u);
  topology.artifacts.emplace("authored_alpha.u8", raw(alpha));topology.artifacts.emplace("authored_clean_scene.rgba8", raw(clean));topology.artifacts.emplace("authored_source.rgba8", raw(origin_pixels_));
  filtered.metadata["independent_authored_truth"] = {{"alpha_artifact", "topology_authored_alpha.u8"}, {"source_artifact", "topology_authored_source.rgba8"}, {"clean_scene_artifact", "topology_authored_clean_scene.rgba8"},
    {"authoring", "Independent opaque original band glyph rectangles over authored scene; graph/M outputs never determine alpha"}, {"target_xy", {qx, qy}}, {"target_alpha", 0u}, {"target_source_clean_max_RGB8_delta", 0u}};
}
