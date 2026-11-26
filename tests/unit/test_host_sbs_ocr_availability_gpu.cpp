/** Production joined-inference regression for an absent optional OCR model. */
#include <gtest/gtest.h>

#ifdef _WIN32
  #include "host_sbs_limiter_oracle.h"
  #include <algorithm>
  #include <bit>
  #include <chrono>
  #include <cstddef>
  #include <cstdint>
  #include <cstdlib>
  #include <cstring>
  #include <filesystem>
  #include <memory>
  #include <optional>
  #include <src/depth_coordinate_v2.h>
  #include <src/host_sbs_completed_subtitle_programs.h>
  #include <src/host_sbs_shader_cache.h>
  #include <src/model_manager.h>
  #include <src/prod_zipdepth_convex2x.h>
  #include <src/video_depth_estimator.h>
  #include <string>
  #include <string_view>
  #include <thread>
  #include <vector>

namespace {
  using Microsoft::WRL::ComPtr;
  namespace v2 = models::depth_coordinate_v2;
  namespace receipt = models::host_sbs_gpu_completion_receipt;

  struct private_assets_t {
    std::filesystem::path path;

    ~private_assets_t() {
      if (!path.empty()) {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
      }
    }

    bool create_without_ocr(const std::filesystem::path &source, std::string &error) {
      // Stay on the packaged models' volume so large ONNX/engine files can be hardlinked.
      // Only a directory created by this instance can reach the cleanup owner above.
      for (unsigned attempt = 0u; attempt < 100u && path.empty(); ++attempt) {
        const auto candidate = source.parent_path() /
                               ("noocr-" + std::to_string(GetCurrentProcessId()) + "-" + std::to_string(attempt));
        std::error_code ec;
        if (std::filesystem::create_directory(candidate, ec)) {
          path = candidate;
        } else if (ec) {
          error = ec.message();
          return false;
        }
      }
      if (path.empty()) {
        error = "Could not reserve private missing-OCR assets";
        return false;
      }
      std::error_code ec;
      std::filesystem::create_directory(path / "shaders", ec);
      if (ec) {
        error = ec.message();
        return false;
      }
      std::filesystem::copy(source / "shaders" / "directx", path / "shaders" / "directx", std::filesystem::copy_options::recursive, ec);
      if (ec) {
        error = ec.message();
        return false;
      }
      const std::string model {models::prod_zipdepth_convex2x::logical_model};
      // The Direct Glyph decoder is packaged apart from the OCR detector; only the detector is withheld.
      const std::string glyph {models::direct_glyph_model_name};
      for (const auto &entry : std::filesystem::directory_iterator(source)) {
        if (!entry.is_regular_file()) {
          continue;
        }
        const auto name = entry.path().filename().string();
        const bool engine = entry.path().extension() == ".engine";
        if (name != model + ".onnx" && !(engine && (name.starts_with(model + ".") || name.starts_with(glyph + ".")))) {
          continue;
        }
        std::filesystem::create_hard_link(entry.path(), path / entry.path().filename(), ec);
        if (ec) {
          error = ec.message();
          return false;
        }
      }
      if (!std::filesystem::is_regular_file(path / (model + ".onnx"))) {
        error = "Missing packaged fused depth model";
        return false;
      }
      const std::filesystem::path glyph_model {models::direct_glyph_model_preset(models::host_sbs_direct_glyph_gpu::tile_profile_e::common, true).asset_path};
      std::filesystem::create_directory(path / glyph_model.parent_path(), ec);
      if (!ec) {
        std::filesystem::create_hard_link(source / glyph_model, path / glyph_model, ec);
      }
      if (ec) {
        error = "Missing packaged Direct Glyph model: " + ec.message();
        return false;
      }
      return !std::filesystem::exists(path / std::filesystem::path {models::ocr_model_asset_path});
    }
  };

  bool read_words(ID3D11Device *device, ID3D11DeviceContext *context, ID3D11ShaderResourceView *view, std::vector<std::uint32_t> &words) {
    if (!view) {
      return false;
    }
    ComPtr<ID3D11Resource> resource;
    view->GetResource(&resource);
    D3D11_RESOURCE_DIMENSION dimension {};
    resource->GetType(&dimension);
    D3D11_MAPPED_SUBRESOURCE mapped {};
    if (dimension == D3D11_RESOURCE_DIMENSION_BUFFER) {
      ComPtr<ID3D11Buffer> buffer, staging;
      if (FAILED(resource.As(&buffer))) {
        return false;
      }
      D3D11_BUFFER_DESC desc {};
      buffer->GetDesc(&desc);
      if (desc.ByteWidth % sizeof(std::uint32_t) != 0u) {
        return false;
      }
      words.resize(desc.ByteWidth / sizeof(std::uint32_t));
      desc.Usage = D3D11_USAGE_STAGING;
      desc.BindFlags = 0u;
      desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      desc.MiscFlags = 0u;
      desc.StructureByteStride = 0u;
      if (FAILED(device->CreateBuffer(&desc, nullptr, &staging))) {
        return false;
      }
      context->CopyResource(staging.Get(), buffer.Get());
      if (FAILED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) {
        return false;
      }
      std::memcpy(words.data(), mapped.pData, words.size() * sizeof(std::uint32_t));
      context->Unmap(staging.Get(), 0u);
      return true;
    }
    if (dimension != D3D11_RESOURCE_DIMENSION_TEXTURE2D) {
      return false;
    }
    ComPtr<ID3D11Texture2D> texture, staging;
    if (FAILED(resource.As(&texture))) {
      return false;
    }
    D3D11_TEXTURE2D_DESC desc {};
    texture->GetDesc(&desc);
    if ((desc.Format != DXGI_FORMAT_R32_FLOAT && desc.Format != DXGI_FORMAT_R32_UINT) || desc.MipLevels != 1u || desc.ArraySize != 1u) {
      return false;
    }
    const auto width = desc.Width, height = desc.Height;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0u;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.MiscFlags = 0u;
    if (FAILED(device->CreateTexture2D(&desc, nullptr, &staging))) {
      return false;
    }
    context->CopyResource(staging.Get(), texture.Get());
    if (FAILED(context->Map(staging.Get(), 0u, D3D11_MAP_READ, 0u, &mapped))) {
      return false;
    }
    words.resize(static_cast<std::size_t>(width) * height);
    for (UINT y = 0u; y < height; ++y) {
      std::memcpy(words.data() + static_cast<std::size_t>(y) * width, static_cast<const std::byte *>(mapped.pData) + static_cast<std::size_t>(y) * mapped.RowPitch, static_cast<std::size_t>(width) * sizeof(std::uint32_t));
    }
    context->Unmap(staging.Get(), 0u);
    return true;
  }

  class TensorRtOcrAvailabilityGpuTest: public testing::Test {
  protected:
    static constexpr UINT width = 1280u, height = 720u;
    private_assets_t assets;
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<ID3D11ShaderResourceView> source;

    void SetUp() override {
      const auto *enabled = std::getenv("APOLLO_RUN_TENSORRT_TESTS");
      if (!enabled || std::string_view {enabled} != "1") {
        GTEST_SKIP() << "Set APOLLO_RUN_TENSORRT_TESTS=1 for the local NVIDIA/TensorRT integration check.";
      }
      std::string error;
      const auto packaged = std::filesystem::path {SUNSHINE_TEST_BIN_DIR}.parent_path() / "assets";
      ASSERT_TRUE(assets.create_without_ocr(packaged, error)) << error;
      ASSERT_FALSE(std::filesystem::exists(assets.path / std::filesystem::path {models::ocr_model_asset_path}));

      D3D_FEATURE_LEVEL actual {};
      constexpr D3D_FEATURE_LEVEL requested[] {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
      ASSERT_TRUE(SUCCEEDED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT, requested, static_cast<UINT>(std::size(requested)), D3D11_SDK_VERSION, &device, &actual, &context)));

      std::vector<std::uint32_t> pixels(static_cast<std::size_t>(width) * height);
      for (UINT y = 0u; y < height; ++y) {
        for (UINT x = 0u; x < width; ++x) {
          const auto r = 32u + 159u * x / (width - 1u), g = 24u + 143u * y / (height - 1u);
          const auto b = 20u + ((x / 64u + y / 48u) % 2u) * 68u;
          pixels[static_cast<std::size_t>(y) * width + x] = 0xff000000u | (r << 16u) | (g << 8u) | b;
        }
      }
      D3D11_TEXTURE2D_DESC desc {};
      desc.Width = width;
      desc.Height = height;
      desc.MipLevels = 1u;
      desc.ArraySize = 1u;
      desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
      desc.SampleDesc.Count = 1u;
      desc.Usage = D3D11_USAGE_IMMUTABLE;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      const D3D11_SUBRESOURCE_DATA initial {pixels.data(), width * sizeof(std::uint32_t), 0u};
      ComPtr<ID3D11Texture2D> texture;
      ASSERT_TRUE(SUCCEEDED(device->CreateTexture2D(&desc, &initial, &texture)));
      ASSERT_TRUE(SUCCEEDED(device->CreateShaderResourceView(texture.Get(), nullptr, &source)));
    }
  };
}  // namespace

TEST_F(TensorRtOcrAvailabilityGpuTest, MissingPackagedOcrPreservesJoinedDepthAndExactBase) {
  std::string error;
  const auto &calibration = v2::model_calibrations.front();
  const config::depth_model_info model {std::string {calibration.depth_model}, std::string {calibration.depth_model_url}};
  using estimator_t = models::video_depth_estimator;
  estimator_t estimator {device, context, assets.path, config::video_t::sbs_t {}, model,
                         estimator_t::normalized_depth_diagnostics_e::enabled,
                         estimator_t::subtitle_legacy_matte_e::on_demand};
  ASSERT_TRUE(estimator.is_valid());
  const auto glyph = models::host_sbs_completed_subtitle_programs::canonical_configuration(assets.path / "shaders" / "directx");
  ASSERT_TRUE(estimator.configure_subtitle_direct_glyph(glyph, error)) << error;

  const auto expect_diagnostic_views = [](const models::depth_observation_views_t &observation, bool expected) {
    EXPECT_FALSE(observation.base_diagnostics_omitted);
    EXPECT_TRUE(observation.shadow_vertical_majorant);
    EXPECT_TRUE(observation.shadow_vertical_conditioned);
    EXPECT_EQ(observation.subtitle_conditioning_diagnostics_current, expected);
    for (auto *view : {
           observation.shadow_conditioned_final_parallax.Get(),
           observation.subtitle_contour_field_pin.Get(),
           observation.subtitle_contour_distance.Get(),
           observation.subtitle_gaussian_weight.Get(),
           observation.subtitle_gaussian_blend.Get(),
           observation.subtitle_gaussian_vertical_conditioned.Get(),
           observation.subtitle_gaussian_control.Get(),
         }) {
      EXPECT_EQ(view != nullptr, expected);
    }
  };
  estimator.set_subtitle_conditioning_diagnostics_always(false);
  EXPECT_FALSE(estimator.subtitle_conditioning_diagnostics_allocated_for_tests());

  // A session may start during USER32 movement, before any ordinary observation allocated
  // a native band. Suppression must use initialized placeholders and still publish real depth.
  const auto initially_suppressed = estimator.estimate_depth(source.Get(), models::input_color_space::srgb,
    0x100000000ull, false, {}, models::depth_optional_work_mode_e::suppress_subtitle,
    {.observation_timestamp_us = 0x200000000ull}, 1u);
  ASSERT_TRUE(initially_suppressed.submission.inference_enqueued);
  ASSERT_FALSE(initially_suppressed.submission.subtitle_ocr_inference_enqueued);
  const auto initial_completed = estimator.finish_pending_depth_for_evaluation();
  ASSERT_FALSE(estimator.has_terminal_failure());
  ASSERT_TRUE(initial_completed.observation.completed_frame_valid);
  EXPECT_EQ(initial_completed.observation.completed_frame_id, 0x100000000ull);
  EXPECT_TRUE(initial_completed.observation.subtitle_work_suppressed);
  ASSERT_TRUE(models::parallax_v2_result_is_authenticated(initial_completed.observation));
  expect_diagnostic_views(initial_completed.observation, false);
  EXPECT_FALSE(estimator.subtitle_conditioning_diagnostics_allocated_for_tests());
  EXPECT_EQ(initial_completed.observation.final_parallax(), initial_completed.observation.shadow_base_final_parallax.Get());
  EXPECT_FALSE(initial_completed.observation.subtitle_glyph_matte);

  // Ordinary live work leaves the diagnostic scratch absent until the first late dump.
  for (std::uint64_t observation = 1u; observation <= 3u; ++observation) {
    const bool dump_snapshot = observation == 2u;
    const bool always_diagnose = observation == 3u;
    const bool diagnostics_expected = dump_snapshot || always_diagnose;
    if (always_diagnose) {
      estimator.set_subtitle_conditioning_diagnostics_always(true);
    }
    const auto frame = 0x100000000ull + observation, clock = 0x200000000ull + observation * 33333u;
    SCOPED_TRACE(frame);
    const auto reusable_deadline = std::chrono::steady_clock::now() + std::chrono::seconds {2};
    while (!estimator.can_accept_frame() && !estimator.has_terminal_failure() &&
           std::chrono::steady_clock::now() < reusable_deadline) {
      std::this_thread::yield();
    }
    ASSERT_TRUE(estimator.can_accept_frame());
    const auto submitted = estimator.estimate_depth(source.Get(), models::input_color_space::srgb, frame, false, {}, models::depth_optional_work_mode_e::ordinary, {.observation_timestamp_us = clock}, 1u);
    ASSERT_TRUE(submitted.submission.inference_enqueued);
    EXPECT_FALSE(submitted.submission.subtitle_ocr_inference_enqueued);
    const auto completed = estimator.finish_pending_depth_for_evaluation(models::input_color_space::srgb, dump_snapshot);
    ASSERT_FALSE(estimator.has_terminal_failure());
    ASSERT_TRUE(completed.observation.completed_frame_valid);
    EXPECT_EQ(completed.observation.completed_frame_id, frame);
    EXPECT_FALSE(completed.observation.subtitle_work_suppressed);
    ASSERT_TRUE(models::parallax_v2_result_is_authenticated(completed.observation));
    EXPECT_EQ(completed.observation.publication.expected_work, models::near_identical_work_subtitle_observation);

    context->Flush();
    std::optional<receipt::receipt_t> publication;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds {2};
    do {
      const auto batch = estimator.poll_publication_receipts();
      ASSERT_FALSE(batch.failed);
      if (batch.current_publication && receipt::matches_publication(*batch.current_publication, completed.observation.publication)) {
        publication = batch.current_publication;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds {1});
    } while (std::chrono::steady_clock::now() < deadline);
    ASSERT_TRUE(publication);
    EXPECT_EQ(publication->depth, receipt::depth_disposition_e::infer);
    EXPECT_TRUE(publication->depth_cache_authorized());
    EXPECT_EQ(publication->depth_owner_frame_id, frame);
    EXPECT_EQ(publication->depth_owner_timestamp_us, clock);
    EXPECT_EQ(publication->subtitle, receipt::subtitle_disposition_e::abstention);
    EXPECT_TRUE(publication->subtitle_is_current());
    EXPECT_EQ(publication->subtitle_frame_id, frame);

    std::vector<std::uint32_t> base, final, ocr, plane, condition;
    ASSERT_TRUE(read_words(device.Get(), context.Get(), completed.observation.shadow_base_final_parallax.Get(), base));
    expect_diagnostic_views(completed.observation, diagnostics_expected);
    EXPECT_EQ(estimator.subtitle_conditioning_diagnostics_allocated_for_tests(), diagnostics_expected);
    if (dump_snapshot) {
      ASSERT_TRUE(completed.observation.raw_model_depth_snapshot);
      ASSERT_TRUE(completed.observation.model_input_snapshot);
    }
    ASSERT_TRUE(read_words(device.Get(), context.Get(), completed.observation.final_parallax(), final));
    EXPECT_EQ(final, base);
    ASSERT_TRUE(read_words(device.Get(), context.Get(), completed.observation.ocr_box_record.Get(), ocr));
    ASSERT_EQ(ocr.size(), v2::subtitle_ocr_record_word_count);
    EXPECT_EQ(ocr[0u], v2::subtitle_ocr_record_schema);
    EXPECT_EQ(ocr[1u], v2::subtitle_ocr_record_tag);
    EXPECT_EQ(ocr[2u], 0u);
    EXPECT_EQ(ocr[3u], 0u);
    EXPECT_EQ(ocr[4u], 0u);
    EXPECT_EQ(models::host_sbs_subtitle_plane::join(ocr.data() + 5u), frame);
    EXPECT_TRUE(std::all_of(ocr.begin() + 16u, ocr.end(), [](auto word) {
      return word == 0u;
    }));
    ASSERT_TRUE(read_words(device.Get(), context.Get(), completed.observation.subtitle_locator_state.Get(), plane));
    ASSERT_EQ(plane.size(), v2::subtitle_plane_state_word_count);
    EXPECT_EQ(plane[v2::subtitle_plane_flags_word], 0u);
    EXPECT_EQ(plane[v2::subtitle_plane_current_slot_mask_word], 0u);
    EXPECT_EQ(models::host_sbs_subtitle_plane::join(plane.data() + v2::subtitle_plane_frame_offset), frame);
    EXPECT_EQ(models::host_sbs_subtitle_plane::join(plane.data() + v2::subtitle_plane_observation_timestamp_offset), clock);
    ASSERT_TRUE(read_words(device.Get(), context.Get(), completed.observation.subtitle_condition_params.Get(), condition));
    ASSERT_EQ(condition.size(), v2::subtitle_condition_param_word_count);
    EXPECT_EQ(condition[v2::subtitle_condition_param_active_word], 0u);
    EXPECT_EQ(condition[v2::subtitle_condition_param_current_count_word], 0u);
    EXPECT_EQ(condition[v2::subtitle_condition_param_target_word], 0u);
    EXPECT_EQ(completed.observation.subtitle_glyph_matte != nullptr, diagnostics_expected);
    for (auto *view : {completed.observation.subtitle_contour_support.Get(), completed.observation.subtitle_ocr_probability.Get(), completed.observation.subtitle_selected_matte.Get()}) {
      std::vector<std::uint32_t> mask;
      ASSERT_TRUE(read_words(device.Get(), context.Get(), view, mask));
      EXPECT_TRUE(std::all_of(mask.begin(), mask.end(), [](auto word) {
        return word == 0u;
      }));
    }
    EXPECT_TRUE(estimator.publication_is_current(completed.observation.publication));
  }

  // Live policy exposes the authenticated Base view directly. A late Dump 3D snapshot of an
  // ordinary observation publishes a distinct conditioned view for just that completion.
  // A snapshot of hard-disabled subtitle work still sees the exact Base view.
  estimator.set_subtitle_conditioning_diagnostics_always(false);
  for (std::uint64_t observation = 4u; observation <= 7u; ++observation) {
    const bool subtitle_disabled = observation == 7u;
    const bool dump_snapshot = observation == 5u || subtitle_disabled;
    const bool diagnostics_expected = dump_snapshot && !subtitle_disabled;
    const auto optional_work = subtitle_disabled ? models::depth_optional_work_mode_e::disable_subtitle :
                                                   models::depth_optional_work_mode_e::ordinary;
    const auto frame = 0x100000000ull + observation, clock = 0x200000000ull + observation * 33333u;
    SCOPED_TRACE(frame);
    const auto reusable_deadline = std::chrono::steady_clock::now() + std::chrono::seconds {2};
    while (!estimator.can_accept_frame() && !estimator.has_terminal_failure() &&
           std::chrono::steady_clock::now() < reusable_deadline) {
      std::this_thread::yield();
    }
    ASSERT_TRUE(estimator.can_accept_frame());
    const auto submitted = estimator.estimate_depth(source.Get(), models::input_color_space::srgb, frame, false, {}, optional_work, {.observation_timestamp_us = clock}, 1u);
    ASSERT_TRUE(submitted.submission.inference_enqueued);
    const auto completed = estimator.finish_pending_depth_for_evaluation(models::input_color_space::srgb, dump_snapshot);
    ASSERT_FALSE(estimator.has_terminal_failure());
    ASSERT_TRUE(completed.observation.completed_frame_valid);
    EXPECT_EQ(completed.observation.completed_frame_id, frame);
    ASSERT_TRUE(models::parallax_v2_result_is_authenticated(completed.observation));
    EXPECT_EQ(completed.observation.subtitle_work_suppressed, subtitle_disabled);
    expect_diagnostic_views(completed.observation, diagnostics_expected);
    EXPECT_EQ(completed.observation.subtitle_glyph_matte != nullptr, diagnostics_expected);
    auto *base_view = completed.observation.shadow_base_final_parallax.Get();
    if (diagnostics_expected) {
      EXPECT_NE(completed.observation.final_parallax(), base_view);
      std::vector<std::uint32_t> base, final;
      ASSERT_TRUE(read_words(device.Get(), context.Get(), base_view, base));
      ASSERT_TRUE(read_words(device.Get(), context.Get(), completed.observation.final_parallax(), final));
      EXPECT_EQ(final, base);
    } else {
      EXPECT_EQ(completed.observation.final_parallax(), base_view);
    }
    if (dump_snapshot) {
      ASSERT_TRUE(completed.observation.raw_model_depth_snapshot);
      ASSERT_TRUE(completed.observation.model_input_snapshot);
    }
    std::vector<std::uint32_t> matte;
    if (!subtitle_disabled) {
      ASSERT_TRUE(read_words(device.Get(), context.Get(), completed.observation.subtitle_selected_matte.Get(), matte));
      EXPECT_TRUE(std::all_of(matte.begin(), matte.end(), [](auto word) {return word == 0u;}));
    }
    if (diagnostics_expected) {
      ASSERT_TRUE(read_words(device.Get(), context.Get(), completed.observation.subtitle_glyph_matte.Get(), matte));
      EXPECT_TRUE(std::all_of(matte.begin(), matte.end(), [](auto word) {return word == 0u;}));
    }
  }
}

TEST_F(TensorRtOcrAvailabilityGpuTest, PaddedLiveBaseRestoresFullDiagnosticsForLateSnapshot) {
  namespace shaders = models::host_sbs_shader_cache;
  const auto shader_sources = shaders::snapshot_sources(assets.path / "shaders" / "directx", shaders::parallax_v2_producer_specs);
  ASSERT_TRUE(shader_sources);
  ASSERT_EQ(shaders::source_closure_sha256(shader_sources), v2::shader_source_closure_sha256);
  for (const auto &spec : {shaders::depth_coordinate_v2_map_compact, shaders::depth_coordinate_v2_vertical_limit_compact, shaders::depth_coordinate_v2_limit_compact}) {
    const auto bytecode = shaders::get_embedded(shader_sources, spec);
    ASSERT_TRUE(bytecode) << spec.filename;
    ComPtr<ID3D11ComputeShader> shader;
    ASSERT_TRUE(SUCCEEDED(device->CreateComputeShader(bytecode->data(), bytecode->size(), nullptr, &shader))) << spec.filename;
  }

  using estimator_t = models::video_depth_estimator;
  const auto &calibration = v2::model_calibrations.front();
  const config::depth_model_info model {std::string {calibration.depth_model}, std::string {calibration.depth_model_url}};
  const auto make_estimator = [&]() {
    auto estimator = std::make_unique<estimator_t>(
      device, context, assets.path, config::video_t::sbs_t {}, model,
      estimator_t::normalized_depth_diagnostics_e::disabled,
      estimator_t::subtitle_legacy_matte_e::on_demand
    );
    estimator->set_subtitle_conditioning_diagnostics_always(false);
    return estimator;
  };

  constexpr models::depth_source_rect_t crop {300u, 100u, 950u, 620u};
  const auto coarse = models::fit_host_sbs_v2_depth_tensor_shape(width, height);
  const auto plan = models::plan_host_sbs_v2_video_region(crop, width, height, coarse);
  const auto high = models::host_sbs_convex2x_field_shape(coarse);
  ASSERT_TRUE(plan);
  ASSERT_TRUE(high);
  const auto content = models::host_sbs_convex2x_content_rect(plan->tensor_content, coarse);
  ASSERT_TRUE(content);
  ASSERT_GT(content->left, 0u);
  ASSERT_LT(content->right, static_cast<std::uint32_t>(high->width));
  ASSERT_GT(content->width(), v2::limiter_serial_max_lines);
  const models::depth_input_region_t region {
    .source_width = width,
    .source_height = height,
    .left = crop.left,
    .top = crop.top,
    .right = crop.right,
    .bottom = crop.bottom,
    .tensor_content = plan->tensor_content,
    .analysis_generation = 1u,
    .authority = models::depth_analysis_authority_e::chromium_video,
  };

  const auto expect_field = [&](ID3D11ShaderResourceView *view, const std::vector<float> &expected, const char *name) {
    std::vector<std::uint32_t> actual;
    ASSERT_TRUE(read_words(device.Get(), context.Get(), view, actual)) << name;
    ASSERT_EQ(actual.size(), expected.size()) << name;
    const auto mismatch = std::mismatch(expected.begin(), expected.end(), actual.begin(), [](float value, std::uint32_t word) {
      return std::bit_cast<std::uint32_t>(value) == word;
    });
    if (mismatch.first != expected.end()) {
      const auto index = mismatch.first - expected.begin();
      ADD_FAILURE() << name << " differs at " << index % high->width << ',' << index / high->width
                    << " expected bits=" << std::bit_cast<std::uint32_t>(*mismatch.first)
                    << " actual bits=" << *mismatch.second;
    }
  };

  const auto complete = [&](estimator_t &target, std::uint64_t frame, bool snapshot, models::depth_step_result_t &result) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds {2};
    while (!target.can_accept_frame() && !target.has_terminal_failure() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
    ASSERT_TRUE(target.can_accept_frame());
    const auto mode = frame <= 3u ? models::depth_optional_work_mode_e::ordinary :
                                    models::depth_optional_work_mode_e::disable_subtitle;
    const auto submitted = target.estimate_depth(source.Get(), models::input_color_space::srgb, frame, false,
      region, mode, {.observation_timestamp_us = frame * 33333u}, 1u);
    ASSERT_TRUE(submitted.submission.inference_enqueued);
    ASSERT_FALSE(submitted.submission.gpu_undecided_transaction_enqueued);
    // The Dump 3D request arrives after submission and before the direct completion.
    result = target.finish_pending_depth_for_evaluation(models::input_color_space::srgb, snapshot);
    ASSERT_FALSE(target.has_terminal_failure());
    ASSERT_TRUE(models::parallax_v2_result_is_authenticated(result.observation));
    ASSERT_EQ(result.observation.completed_frame_id, frame);
  };

  constexpr std::uint64_t frame_count = 6u;
  // Collect diagnostics with a separate camera history and release the reference estimator
  // before creating the live estimator. Only one TensorRT context is admitted per model.
  std::vector<std::vector<float>> reference_candidates(frame_count);
  {
    const auto full_grid = make_estimator();
    ASSERT_TRUE(full_grid->is_valid());
    for (std::uint64_t frame = 1u; frame <= frame_count; ++frame) {
      SCOPED_TRACE(frame);
      models::depth_step_result_t reference;
      ASSERT_NO_FATAL_FAILURE(complete(*full_grid, frame, true, reference));
      std::vector<std::uint32_t> words;
      ASSERT_TRUE(read_words(device.Get(), context.Get(), reference.observation.shadow_candidate_parallax.Get(), words));
      auto &candidate = reference_candidates[frame - 1u];
      candidate.resize(words.size());
      std::transform(words.begin(), words.end(), candidate.begin(), std::bit_cast<float, std::uint32_t>);
    }
  }

  const auto estimator = make_estimator();
  ASSERT_TRUE(estimator->is_valid());
  for (std::uint64_t frame = 1u; frame <= frame_count; ++frame) {
    SCOPED_TRACE(frame);
    const bool snapshot = frame % 3u == 2u;
    models::depth_step_result_t completed;
    ASSERT_NO_FATAL_FAILURE(complete(*estimator, frame, snapshot, completed));
    const auto &observation = completed.observation;
    ASSERT_FALSE(observation.gpu_undecided_completion);
    ASSERT_EQ(observation.field_content, *content);
    ASSERT_EQ(observation.field_width, high->width);
    ASSERT_EQ(observation.field_height, high->height);
    ASSERT_EQ(observation.base_diagnostics_omitted, !snapshot);
    ASSERT_EQ(static_cast<bool>(observation.shadow_candidate_parallax), snapshot);
    ASSERT_EQ(static_cast<bool>(observation.shadow_vertical_majorant), snapshot);
    ASSERT_EQ(static_cast<bool>(observation.shadow_vertical_conditioned), snapshot);
    ASSERT_EQ(static_cast<bool>(observation.raw_model_depth_snapshot), snapshot);
    ASSERT_EQ(static_cast<bool>(observation.model_input_snapshot), snapshot);

    const auto &candidate = reference_candidates[frame - 1u];
    const auto vertical = host_sbs_test::exact_q30_vertical_oracle(candidate, high->width, high->height, content->width());
    ASSERT_EQ(vertical.majorant.size(), candidate.size());
    ASSERT_EQ(vertical.conditioned.size(), candidate.size());
    const auto base = host_sbs_test::exact_q30_horizontal_oracle(vertical.conditioned, high->width, high->height, content->width());
    ASSERT_NO_FATAL_FAILURE(expect_field(observation.shadow_base_final_parallax.Get(), base, "Base"));
    if (snapshot) {
      ASSERT_NO_FATAL_FAILURE(expect_field(observation.shadow_candidate_parallax.Get(), candidate, "candidate"));
      ASSERT_NO_FATAL_FAILURE(expect_field(observation.shadow_vertical_majorant.Get(), vertical.majorant, "vertical majorant"));
      ASSERT_NO_FATAL_FAILURE(expect_field(observation.shadow_vertical_conditioned.Get(), vertical.conditioned, "vertical conditioned"));
    }
  }
}
#endif
