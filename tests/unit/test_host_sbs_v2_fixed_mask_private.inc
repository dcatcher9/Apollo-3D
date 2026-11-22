// Private opt-in experiment, included by test_host_sbs_v2_live_warp.cpp only.
// It never publishes captured provenance as a current producer receipt.
TEST(HostSbsV2LiveWarpGpuTest, PrivateSharedUiFixedPlaneMaskComparison) {
  const auto *input_env = std::getenv("SUNSHINE_TEST_SHARED_UI_FIXED_MASK_INPUTS");
  const auto *plan_env = std::getenv("SUNSHINE_TEST_SHARED_UI_FIXED_MASK_METRIC_PLAN");
  const auto *output_env = std::getenv("SUNSHINE_TEST_SHARED_UI_FIXED_MASK_OUTPUT");
  if (!input_env || !*input_env || !plan_env || !*plan_env || !output_env || !*output_env) {
    GTEST_SKIP() << "Explicit fixed-mask input, frozen metric plan and NEW output required";
  }
  namespace fs = std::filesystem;
  using json = nlohmann::json;
  try {
    const auto require = [](bool value, const std::string &message) {
      if (!value) {
        throw std::runtime_error(message);
      }
    };
    const auto read = [&](const fs::path &path) {
      std::ifstream stream(path, std::ios::binary | std::ios::ate);
      require(bool(stream) && stream.tellg() >= 0, "Cannot read fixed-mask input: " + path.string());
      std::vector<std::byte> bytes(static_cast<std::size_t>(stream.tellg()));
      stream.seekg(0);
      stream.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
      require(bool(stream), "Incomplete fixed-mask input: " + path.string());
      return bytes;
    };
    const auto sha = [](const auto &bytes) {
      const auto hash = crypto::hash(std::string_view(reinterpret_cast<const char *>(bytes.data()), bytes.size()));
      constexpr char digits[] = "0123456789abcdef";
      std::string value;
      for (auto byte : hash) {
        value.push_back(digits[byte >> 4u]);
        value.push_back(digits[byte & 15u]);
      }
      return value;
    };
    const auto file_binding = [&](const fs::path &path) {
      const auto bytes = read(path);
      return json {{"path", path.string()}, {"byte_count", bytes.size()}, {"sha256", sha(bytes)}};
    };
    const auto document = [&](const auto &bytes) {
      return json::parse(std::string_view(reinterpret_cast<const char *>(bytes.data()), bytes.size()));
    };
    const auto input_bytes = read(input_env), plan_bytes = read(plan_env);
    const auto spec = document(input_bytes), plan = document(plan_bytes);
    const bool subset = spec.at("schema") == 2u && spec.at("role") == "private-fixed-plane-mask-subset-known-layers";
    const bool additive = spec.at("schema") == 3u && spec.at("role") == "private-fixed-plane-mask-superset-known-layers";
    require((subset || additive) && spec.at("development_only") == true && spec.at("cases").size() == 3u && spec.at("candidate_status") == "BOUND_ACTUAL_GPU_PRIVATE_CANDIDATE_PENDING_RENDER_QUALITY" && spec.at("candidate_disposition") == "experimental-unqualified", "Fixed-mask inputs must bind an actual GPU development candidate, never an oracle or rejected plumbing mask");
    require((additive ? plan.at("schema") == 3u && plan.at("status") == "PREDECLARED_BEFORE_ADDITIVE_FIXED_PLANE_RENDER_OUTPUTS" : plan.at("schema") == 2u && plan.at("status") == "PREDECLARED_BEFORE_FIXED_PLANE_MASK_RENDER_OUTPUTS") && plan.contains("native_vertical_step_diagnostic") && plan.at("runtime") == spec.at("runtime") && plan.at("base_layer_inputs") == spec.at("base_layer_inputs") && plan.at("cases").size() == 3u, "Fixed-mask metric plan/runtime/base specification mismatch");
    const auto require_bound_bytes = [&](const json &binding, const auto &bytes) {
      require(binding.at("byte_count") == bytes.size() && binding.at("sha256") == sha(bytes), "Additive fixed-mask decoded bytes differ from their exact binding");
    };
    const auto base_binding = spec.at("base_layer_inputs");
    require(file_binding(base_binding.at("path").get<std::string>()) == base_binding, "Fixed-mask original layer specification changed");
    const auto base_bytes = read(base_binding.at("path").get<std::string>());
    const auto base_spec = document(base_bytes);
    const auto &reference_binding = spec.at("reference_specification");
    const auto reference_path = reference_binding.at("path").get<std::string>();
    require(file_binding(reference_path) == reference_binding, "Fixed-mask independent layer shader specification changed");
    const auto reference_bytes = read(reference_path);
    const auto reference = document(reference_bytes);
    const auto &evidence = spec.at("candidate_source_evidence");
    require(evidence.at("alpha_truth_used_to_build_candidate") == false, "Candidate may not be constructed from metric truth");
    if (additive) {
      require_bound_bytes(base_binding, base_bytes);require_bound_bytes(reference_binding, reference_bytes);
      const auto mode = spec.at("mode").get<std::string>();
      require(spec.at("candidate_admission_rule") == "superset" && plan.at("candidate_admission_rule") == "superset" && plan.at("mode") == mode && plan.at("candidate_source_evidence") == evidence && base_spec.at("cases").size() == 3u, "Additive fixed-mask plan must bind the same ordered3 inputs and actual candidate evidence");
      require((mode == "overlap" && evidence.at("diagnostics_key") == "private_overlap_width_diagnostics") || (mode == "soft" && evidence.at("diagnostics_key") == "private_soft_layer_diagnostics"), "Additive fixed-mask evidence schema must explicitly match its mode");
      require(evidence.at("source_closure_sha256").get<std::string>().size() == 64u, "Additive fixed-mask source closure must be explicit");
    }
    const auto candidate_path = evidence.at("native_report").at("path").get<std::string>();
    const auto job_path = evidence.at("native_job").at("path").get<std::string>();
    require(file_binding(candidate_path) == evidence.at("native_report") && file_binding(job_path) == evidence.at("native_job"), "Fixed-mask candidate report/job changed");
    const auto candidate_bytes = read(candidate_path), job_bytes = read(job_path);
    const auto candidate_report = document(candidate_bytes), candidate_job = document(job_bytes);
    if (additive) {require_bound_bytes(evidence.at("native_report"), candidate_bytes);require_bound_bytes(evidence.at("native_job"), job_bytes);}
    require(candidate_job.at("exit_code") == 0 && candidate_job.at("test_unchanged") == true && candidate_job.at("fixture_unchanged") == true && candidate_job.at("runtime") == spec.at("runtime") && candidate_job.at("test").at("sha256") == evidence.at("native_executable").at("sha256") && candidate_report.at("test_executable").at("sha256") == evidence.at("native_executable").at("sha256") && file_binding(evidence.at("native_executable").at("path").get<std::string>()) == evidence.at("native_executable"), "Candidate must join its unchanged actual native executable/job");
    require(candidate_report.at("cases").size() == 3u, "Candidate report is development3 only");
    const fs::path output(output_env);
    require(!fs::exists(output), "Fixed-mask output must be new");
    fs::create_directories(output);
    live_v2_warp_fixture_t fixture;
    std::string error;
    ASSERT_TRUE(fixture.initialize(error, warp_shader_e::selected_outline, false, true)) << error;
    json report {{"schema", additive ? 3u : 1u}, {"role", additive ? "private-actual-GPU-fixed-plane-mask-superset-comparison" : "private-actual-GPU-fixed-plane-mask-subset-comparison"}, {"runtime", spec.at("runtime")}, {"not_production_qualified", true}, {"temporal_controller_simulated", false}, {"adapter", fixture.adapter_identity()}, {"input_specification", file_binding(input_env)}, {"metric_plan", file_binding(plan_env)}, {"reference_specification", reference_binding}, {"production_shared_warp_closure", models::host_sbs_shared_ui_warp::closure}, {"fixture_source", file_binding(fs::path(SUNSHINE_SOURCE_DIR) / "tests/unit/test_host_sbs_v2_live_warp.cpp")}, {"fixture_include", file_binding(fs::path(SUNSHINE_SOURCE_DIR) / "tests/unit/test_host_sbs_v2_fixed_mask_private.inc")}, {"scope", "Static development3, source2. Same actual source/Base/state/b2/b3/fixed plane; each selected mask has a new authored frame/SCT/receipt. Captured producer receipts are never promoted."}, {"cases", json::array()}};
    if (additive) {report["mode"] = spec.at("mode");report["candidate_admission_rule"] = "superset";report["candidate_source_evidence"] = evidence;report["quality_accepted"] = false;}
    std::array<std::string, 3u> ordered_names;
    for (std::size_t index = 0u; index < 3u; ++index) {
      const auto &entry = spec.at("cases").at(index);
      const auto &original = base_spec.at("cases").at(index);
      const auto name = entry.at("name").get<std::string>();
      SCOPED_TRACE(name);
      require((additive ? !name.empty() : name == "latin-bold" || name == "cjk" || name == "warm-ink") && entry.at("split") == "development" && entry.at("frame") == 2u && original.at("name") == name && plan.at("cases").at(index).at("name") == name && entry.at("fixed_plane_bits") == plan.at("cases").at(index).at("fixed_plane_bits"), "Fixed-mask development case/plane ordering");
      if (additive) {
        for (std::size_t previous = 0u; previous < index; ++previous) require(ordered_names[previous] != name, "Additive fixed-mask names must be distinct within the exact ordered3");
        ordered_names[index] = name;
        require(entry.at("candidate_provenance").at("native_case_name") == plan.at("cases").at(index).at("native_case_name"), "Additive fixed-mask native entry must match the predeclared ordering");
      }
      const auto source_shape = entry.at("source").get<std::array<UINT, 2u>>();
      const auto field_shape = entry.at("field").get<std::array<UINT, 2u>>();
      const auto rect = entry.at("analysis_rect").get<std::array<UINT, 4u>>();
      const auto crop = entry.at("native_crop").get<std::array<UINT, 3u>>();
      const UINT width = source_shape[0u], height = source_shape[1u];
      require(source_shape == std::array<UINT, 2u> {1920u, 1080u} && rect == std::array<UINT, 4u> {0u, 0u, width, height} && crop == std::array<UINT, 3u> {width, 760u, 320u} && field_shape[0u] > 0u && field_shape[0u] <= 5120u && field_shape[1u] > 0u && field_shape[1u] <= 5120u, "Fixed-mask bounded full-frame development geometry");
      json bindings = json::object();
      std::map<std::string, std::vector<std::byte>> imported;
      const auto load_bound = [&]<typename T>(const char *key, const json &bound, std::size_t count) {
        const auto bytes = read(bound.at("path").get<std::string>());
        require(bytes.size() == count * sizeof(T) && bound.at("byte_count") == bytes.size() && bound.at("sha256") == sha(bytes), std::string("Fixed-mask immutable operand binding: ") + key);
        bindings[key] = bound;
        imported.emplace(key, bytes);
        std::vector<T> values(count);
        std::memcpy(values.data(), bytes.data(), bytes.size());
        return values;
      };
      const auto load = [&]<typename T>(const char *key, std::size_t count) {
        require(entry.at(key) == original.at(key), std::string("Candidate changed frozen source operand: ") + key);
        return load_bound.template operator()<T>(key, entry.at(key), count);
      };
      const auto source = load.template operator()<bgra8_t>("source_bgra8", std::size_t(width) * height);
      const auto background = load.template operator()<bgra8_t>("background_bgra8", source.size());
      const auto foreground = load.template operator()<std::array<std::uint8_t, 4u>>("foreground_rgba8", source.size());
      const auto body = load.template operator()<std::uint8_t>("effective_body_alpha_u8", source.size());
      const auto raw_body = load.template operator()<std::uint8_t>("raw_inner_label_u8", source.size());
      const auto base = load.template operator()<float>("base", std::size_t(field_shape[0u]) * field_shape[1u]);
      const auto state_words = load.template operator()<std::uint32_t>("state", 28u);
      const auto snapshot = load.template operator()<rgba16_bits_t>("snapshot", std::size_t(crop[0u]) * crop[2u]);
      const auto geometry_bytes = load.template operator()<std::byte>("geometry", sizeof(host_sbs_v2_geometry_t));
      const auto b3_bytes = load.template operator()<std::byte>("subtitle_constants", sizeof(models::host_sbs_subtitle_render_constants_t));
      const auto control_m = load.template operator()<std::uint32_t>("control_matte", snapshot.size());
      const auto candidate_m = load_bound.template operator()<std::uint32_t>("candidate_matte", entry.at("candidate_matte"), snapshot.size());
      const auto condition_words = load_bound.template operator()<std::uint32_t>("captured_condition", entry.at("provenance").at("captured_condition"), 6u);
      const auto sct_words = load_bound.template operator()<std::uint32_t>("captured_sct", entry.at("provenance").at("captured_sct"), 32u);
      models::depth_coordinate_v2::state_words_t state;
      std::memcpy(state.data(), state_words.data(), sizeof(state));
      host_sbs_v2_geometry_t geometry;
      std::memcpy(&geometry, geometry_bytes.data(), sizeof(geometry));
      subtitle_inputs_t captured;
      std::memcpy(&captured.constants, b3_bytes.data(), sizeof(captured.constants));
      std::copy(condition_words.begin(), condition_words.end(), captured.condition.begin());
      std::copy(sct_words.begin(), sct_words.end(), captured.identity.begin());
      captured.width = crop[0u];
      captured.height = crop[2u];
      captured.matte = control_m;
      captured.snapshot = snapshot;
      const auto plane_bits = entry.at("fixed_plane_bits").get<std::uint32_t>();
      require(captured.condition[5u] == plane_bits && geometry.video_roi_active == 0.0f && captured.constants.enabled == 1u && captured.constants.source_width == width && captured.constants.source_height == height && captured.constants.source_left == 0u && captured.constants.source_top == 0u && captured.identity[10u] == crop[1u] && captured.identity[11u] == crop[2u], "Fixed-mask captured plane/b2/b3/native crop join");
      for (std::size_t pixel = 0u; pixel < source.size(); ++pixel) {
        require(body[pixel] == std::min(raw_body[pixel], foreground[pixel][3u]), "Effective inner metric must preserve original raw truth separately");
      }
      for (std::size_t pixel = 0u; pixel < control_m.size(); ++pixel) {
        if (additive) require(control_m[pixel] <= 1u && candidate_m[pixel] <= 1u && candidate_m[pixel] >= control_m[pixel], "Actual additive candidate must be a literal binary control-mask superset");
        else require(control_m[pixel] <= 1u && candidate_m[pixel] <= control_m[pixel], "Actual candidate must be a literal binary control-mask subset");
      }
      const auto &produced = candidate_report.at("cases").at(index);
      const auto &provenance = entry.at("candidate_provenance");
      const auto artifact_name = evidence.at("artifact_name").get<std::string>();
      const auto &candidate_diagnostic = produced.at(evidence.at("diagnostics_key").get<std::string>());
      const auto &parity = produced.at("expected_production_parity");
      bool diagnostic_inputs_unchanged;
      if (additive) {
        diagnostic_inputs_unchanged = evidence.at("diagnostics_key") == "private_overlap_width_diagnostics" ? candidate_diagnostic.at("source_inputs_and_prior_byteexact_after_consumption") == true :
          candidate_diagnostic.at("bound_inputs_byteexact_after_probe") == true && candidate_diagnostic.at("source_closure_unchanged_after_probe") == true;
        require(candidate_diagnostic.at("source_closure") == evidence.at("source_closure_sha256"), "Additive fixed-mask actual candidate source closure mismatch");
      } else diagnostic_inputs_unchanged = candidate_diagnostic.at("production_inputs_byteexact_after_probe") == true;
      require((additive ? produced.at("name") == provenance.at("native_case_name") : produced.at("name") == "font-challenge-" + name + "-first-cue-current") && produced.at("current_outline_authorized") == true && produced.at("selected_matte_bitexact") == true && parity.size() == 4u && std::all_of(parity.begin(), parity.end(), [](const json &value) {
                return value == true;
              }) &&
                diagnostic_inputs_unchanged && candidate_diagnostic.at("candidate_is_production_mask") == false && produced.at("artifacts").at(artifact_name) == entry.at("candidate_matte") && produced.at("artifacts").at("selected_M.u32").at("sha256") == entry.at("control_matte").at("sha256") && provenance.at("actual_GPU_artifact") == true && provenance.at("not_current_producer_authority") == true && provenance.at("production_parity_required_and_passed") == true,
              "Selected mask must be exact actual GPU candidate artifact, not truth-derived support");
      const auto source5 = load_bound.template operator()<std::array<std::uint8_t, 4u>>("candidate_source5_reference_RGBA8", produced.at("artifacts").at("source5_reference_band_RGBA8.bin"), snapshot.size());
      for (std::size_t pixel = 0u; pixel < source5.size(); ++pixel) {
        const auto &expected = source[std::size_t(crop[1u]) * width + pixel];
        require(source5[pixel] == std::array<std::uint8_t, 4u> {expected.r, expected.g, expected.b, 255u}, "Candidate model source5 must equal pinned original source bytes");
      }
      const fs::path folder = output / ("case_" + std::to_string(index));
      fs::create_directory(folder);
      const auto save = [&](const std::string &filename, const void *data, std::size_t size) {
        const auto path = folder / filename;
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(size));
        stream.close();
        require(bool(stream), "Cannot save fixed-mask artifact");
        return file_binding(path);
      };
      const auto save_vector = [&](const std::string &filename, const auto &values) {
        return save(filename, values.data(), values.size() * sizeof(typename std::decay_t<decltype(values)>::value_type));
      };
      const auto authored_context = [&](UINT route) {
        auto ui = captured;
        // Every captured source fact is unchanged except this new authored frame.
        ui.identity[4u] = 0x74000000u + static_cast<UINT>(index) * 8u + route + 1u;
        ui.identity[5u] = 0u;
        require(ui.identity[4u] != captured.identity[4u] || ui.identity[5u] != captured.identity[5u], "Authored frame must be distinct from captured frame");
        return ui;
      };
      const auto selected_context = [&](const subtitle_inputs_t &ui, const std::vector<std::uint32_t> &selected) {
        auto mask = make_current_outline_inputs(ui);
        mask.matte = selected;
        mask.receipt[60u] = ui.identity[4u];
        mask.receipt[61u] = 0x464d4153u;
        mask.production_shared_ui_pin = true;
        return mask;
      };
      const auto save_pin = [&](const std::string &prefix, const shared_ui_pin_diagnostics_t &pin) {
        return json {{"shape", pin.shape}, {"selected_receipt_authorized", pin.selected_receipt_authorized}, {"field", save_vector(prefix + "native-field.f32", pin.field)}, {"authorized_weights", save_vector(prefix + "authorized-weights.f32", pin.authorized_weights)}, {"active_rows", save_vector(prefix + "active-rows.u32", pin.active_rows)}, {"row_header", save_vector(prefix + "row-header.u32", pin.selected_receipt_header)}};
      };
      json result {{"name", name}, {"source", source_shape}, {"field", field_shape}, {"fixed_plane_bits", plane_bits}, {"input_bindings", bindings}, {"common_state", save("common-state.u32", state.data(), sizeof(state))}, {"common_geometry", save("common-geometry.b2", &geometry, sizeof(geometry))}, {"common_subtitle_constants", save("common-subtitle.b3", &captured.constants, sizeof(captured.constants))}, {"routes", json::object()}};
      auto reference_ui = authored_context(2u);
      std::vector<std::byte> constant_color;
      const std::vector<float> constant_base(base.size(), std::bit_cast<float>(plane_bits));
      ASSERT_TRUE(fixture.render(DXGI_FORMAT_B8G8R8A8_UNORM, 4u, width, height, source.data(), constant_base, constant_base, state, {}, constant_color, error, field_shape[0u], field_shape[1u], geometry)) << error;
      live_v2_warp_fixture_t::private_layer_output_t layered;
      ASSERT_TRUE(fixture.render_private_layer_reference(width, height, background, foreground, body, base, state, field_shape[0u], field_shape[1u], geometry, reference_ui, reference.at("private_layer_shader_root").get<std::string>(), reference.at("private_layer_source_closure_sha256").get<std::string>(), layered, error)) << error;
      require(layered.reconstructed_rgba8.size() == source.size() * 4u, "Fixed-mask source reconstruction shape");
      UINT maximum_error = 0u;
      for (std::size_t pixel = 0u; pixel < source.size(); ++pixel) {
        const std::array<std::uint8_t, 3u> expected {source[pixel].r, source[pixel].g, source[pixel].b};
        for (UINT channel = 0u; channel < 3u; ++channel) {
          maximum_error = std::max(maximum_error, UINT(std::abs(int(std::to_integer<unsigned int>(layered.reconstructed_rgba8[pixel * 4u + channel])) - int(expected[channel]))));
        }
      }
      ASSERT_LE(maximum_error, 1u) << "Independent original layers must reconstruct exact immutable source";
      const auto premultiplied = unpack_rgba32f(layered.shifted_foreground_rgba32f);
      std::vector<float> total_alpha;
      total_alpha.reserve(premultiplied.size());
      for (const auto &pixel : premultiplied) {
        total_alpha.push_back(pixel.a);
      }
      result["reference"] = {{"clean_base_background", save_vector("reference.clean-base.bgra8", layered.base_background_bgra8)}, {"constant_original_source", save_vector("reference.constant-original-source.bgra8", constant_color)}, {"layered_color", save_vector("reference.layered.rgba8", layered.reference_rgba8)}, {"foreground_alpha", save_vector("reference.total-alpha.f32", total_alpha)}, {"body_alpha", save_vector("reference.body-alpha.f32", layered.shifted_body_alpha_f32)}, {"source_reconstruction", save_vector("reference.source-reconstruction.rgba8", layered.reconstructed_rgba8)}, {"source_reconstruction_max_RGB8_error", maximum_error}, {"color_domain", "SDR UNORM numeric encoded RGB, same as production SourceColor"}};
      const std::array<std::pair<const char *, const std::vector<std::uint32_t> *>, 3u> routes {{{"control_mask", &control_m}, {"candidate_mask", &candidate_m}, {"empty_mask_native_Base_diagnostic", nullptr}}};
      const std::vector<std::uint32_t> empty_m(control_m.size(), 0u);
      for (std::size_t route = 0u; route < routes.size(); ++route) {
        const auto &[route_name, selected] = routes[route];
        SCOPED_TRACE(route_name);
        auto ui = authored_context(static_cast<UINT>(route) + 3u);
        auto mask = selected_context(ui, selected ? *selected : empty_m);
        shared_ui_pin_diagnostics_t pin;
        mask.shared_ui_pin_diagnostic_output = &pin;
        std::vector<std::byte> pixels;
        ASSERT_TRUE(fixture.render(DXGI_FORMAT_B8G8R8A8_UNORM, 4u, width, height, source.data(), base, base, state, {}, pixels, error, field_shape[0u], field_shape[1u], geometry, &ui, nullptr, true, &mask)) << error;
        require(pin.selected_receipt_authorized && pin.selected_receipt_header[0u] == 1u, "Each authored selected receipt must authenticate");
        const std::string prefix = std::string(route_name) + ".";
        auto pin_binding = save_pin(prefix, pin);
        if (!selected) {
          require(std::all_of(pin.authorized_weights.begin(), pin.authorized_weights.end(), [](float value) {
                    return value == 0.0f;
                  }) &&
                    std::all_of(pin.active_rows.begin(), pin.active_rows.end(), [](auto value) {
                      return value == 0u;
                    }),
                  "Empty selected M diagnostic must initialize only native Base without active UI rows");
          pin_binding["scope"] = "empty-authored-mask-native-Base-field-only-not-candidate";
          result["native_Base_diagnostic"] = std::move(pin_binding);
        } else {
          result["routes"][route_name] = {{"plane_bits", plane_bits}, {"context_role", "fresh-authored-not-captured-or-current-producer"}, {"production_shared_ui_warp", true}, {"authored_context", {{"condition", save(prefix + "condition.u32", ui.condition.data(), sizeof(ui.condition))}, {"identity", save(prefix + "identity.u32", ui.identity.data(), sizeof(ui.identity))}, {"selected_receipt", save(prefix + "selected-receipt.u32", mask.receipt.data(), sizeof(mask.receipt))}, {"subtitle_constants", result.at("common_subtitle_constants")}}}, {"color", save_vector(prefix + "color.bgra8", pixels)}, {"shared_warp", std::move(pin_binding)}};
        }
      }
      for (const auto &[key, bytes] : imported) {
        require(read(bindings.at(key).at("path").get<std::string>()) == bytes, "Fixed-mask operand changed during actual GPU rendering");
      }
      report["cases"].push_back(std::move(result));
    }
    require(read(input_env) == input_bytes && read(plan_env) == plan_bytes && read(reference_path) == reference_bytes && read(candidate_path) == candidate_bytes && read(job_path) == job_bytes, "Fixed-mask specification/candidate source changed during comparison");
    if (additive) {
      require(read(base_binding.at("path").get<std::string>()) == base_bytes && file_binding(evidence.at("native_executable").at("path").get<std::string>()) == evidence.at("native_executable") &&
        file_binding(report.at("fixture_source").at("path").get<std::string>()) == report.at("fixture_source") && file_binding(report.at("fixture_include").at("path").get<std::string>()) == report.at("fixture_include"), "Additive fixed-mask source/fixture changed during actual GPU comparison");
      report["source_and_ordered_entries_byteexact_after_render"] = true;
    }
    const auto text = report.dump(2);
    std::ofstream stream(output / "report.json", std::ios::binary);
    stream.write(text.data(), static_cast<std::streamsize>(text.size()));
    stream.close();
    require(bool(stream), "Cannot save fixed-plane mask comparison report");
  } catch (const std::exception &failure) {
    FAIL() << failure.what();
  }
}
