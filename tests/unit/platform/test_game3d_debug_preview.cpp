// SPDX-License-Identifier: GPL-3.0-only
#include "src/platform/windows/game3d_debug_preview.h"
#include "src/game3d_debug_protocol.h"
#include "src/game3d_debug_ui_resources.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <wincodec.h>
#include <windows.h>
#include <wrl/client.h>

namespace preview = platf::game3d_debug::preview;
using preview::detail::rgb;
using json = nlohmann::json;

namespace {
  template<class T>
  std::vector<std::uint8_t> bytes(std::initializer_list<T> values) {
    std::vector<std::uint8_t> out(values.size() * sizeof(T));
    std::memcpy(out.data(), values.begin(), out.size());
    return out;
  }

  struct preview_directory {
    std::filesystem::path path = std::filesystem::temp_directory_path() / ("sunshine-preview-test-" + std::to_string(GetCurrentProcessId()) + "-" + std::to_string(GetTickCount64()));

    preview_directory() {
      std::filesystem::create_directories(path);
    }

    ~preview_directory() {
      std::error_code error;
      std::filesystem::remove_all(path, error);
    }
  };

  std::vector<std::uint8_t> read_png(const std::filesystem::path &path) {
    const auto initialized = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    Microsoft::WRL::ComPtr<IWICImagingFactory> factory;
    Microsoft::WRL::ComPtr<IWICBitmapDecoder> decoder;
    Microsoft::WRL::ComPtr<IWICBitmapFrameDecode> frame;
    Microsoft::WRL::ComPtr<IWICFormatConverter> converter;
    unsigned width = 0, height = 0;
    const bool ok = SUCCEEDED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory))) &&
      SUCCEEDED(factory->CreateDecoderFromFilename(path.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnLoad, &decoder)) &&
      SUCCEEDED(decoder->GetFrame(0, &frame)) && SUCCEEDED(frame->GetSize(&width, &height)) &&
      SUCCEEDED(factory->CreateFormatConverter(&converter)) &&
      SUCCEEDED(converter->Initialize(frame.Get(), GUID_WICPixelFormat24bppRGB, WICBitmapDitherTypeNone, nullptr, 0, WICBitmapPaletteTypeCustom));
    std::vector<std::uint8_t> pixels(width * height * 3);
    const bool copied = ok && SUCCEEDED(converter->CopyPixels(nullptr, width * 3, static_cast<UINT>(pixels.size()), pixels.data()));
    converter.Reset(); frame.Reset(); decoder.Reset(); factory.Reset();
    if (SUCCEEDED(initialized)) CoUninitialize();
    if (!copied) throw std::runtime_error("Cannot decode test preview PNG");
    return pixels;
  }
}  // namespace

TEST(Game3dDebugPreview, DecodesNativeFormatsAndPreservesSdr) {
  auto rgba = bytes<std::uint8_t>({8, 127, 244, 0});
  EXPECT_EQ(preview::detail::display_color(preview::detail::sample({"source_color", 1, 1, 28, rgba}, 0, 0), 1), (rgb {8, 127, 244}));
  EXPECT_EQ(preview::detail::display_color(preview::detail::sample({"source_color", 1, 1, 87, rgba}, 0, 0), 1), (rgb {244, 127, 8}));
  auto packed = bytes<std::uint32_t>({1023u | (512u << 10) | (3u << 30)});
  EXPECT_EQ(preview::detail::display_color(preview::detail::sample({"source_color", 1, 1, 24, packed}, 0, 0), 1), (rgb {255, 128, 0}));
  auto fp16 = bytes<std::uint16_t>({0x3c00, 0x3800, 0xbc00, 0x7c00});
  EXPECT_EQ(preview::detail::sample({"source_color", 1, 1, 10, fp16}, 0, 0), (std::array<float, 4> {1, 0.5f, -1, INFINITY}));
}

TEST(Game3dDebugPreview, HdrTonemapUsesCorrectTransferAndPrimaries) {
  EXPECT_EQ(preview::detail::display_color({1, 1, 1, 1}, 2), (rgb {188, 188, 188}));
  // ST.2084 code value for 80 nits: should map to the same preview white as scRGB 1.
  const double p = std::pow(80.0 / 10000.0, 0.1593017578125);
  const float pq80 = static_cast<float>(std::pow((0.8359375 + 18.8515625 * p) / (1 + 18.6875 * p), 78.84375));
  const auto neutral = preview::detail::display_color({pq80, pq80, pq80, 1}, 3);
  for (const auto c : neutral) {
    EXPECT_NEAR(c, 188, 1);
  }
  const auto red = preview::detail::display_color({pq80, 0, 0, 1}, 3);
  EXPECT_GT(red[0], 188);
  EXPECT_EQ(red[1], 0);
  EXPECT_EQ(red[2], 0);
  EXPECT_EQ(preview::detail::display_color({-1, 0, 1, 1}, 2), (rgb {0, 0, 188}));
}

TEST(Game3dDebugPreview, DepthHandlesConstantAndNonfiniteWithoutHidingThem) {
  EXPECT_EQ(preview::detail::scalar_color(4, 4, 4), (rgb {128, 128, 128}));
  EXPECT_EQ(preview::detail::scalar_color(2, 2, 10), (rgb {0, 0, 0}));
  EXPECT_EQ(preview::detail::scalar_color(10, 2, 10), (rgb {255, 255, 255}));
  EXPECT_EQ(preview::detail::scalar_color(NAN, 2, 10), (rgb {255, 0, 255}));
  EXPECT_EQ(preview::detail::display_color({INFINITY, 0, 0, 1}, 2), (rgb {255, 0, 255}));
}

TEST(Game3dDebugPreview, SignedScaleKeepsZeroNeutralAndBothSignsDistinct) {
  EXPECT_EQ(preview::detail::signed_color(0, 1), preview::detail::signed_color(0, 0));
  const auto negative = preview::detail::signed_color(-1, 1), positive = preview::detail::signed_color(1, 1);
  EXPECT_GT(negative[2], negative[0]);
  EXPECT_GT(positive[0], positive[2]);
  EXPECT_EQ(preview::detail::signed_color(INFINITY, 1), (rgb {255, 0, 255}));
}

TEST(Game3dDebugPreview, CropAndJitterExcludeAllocationPadding) {
  const std::array<float, 4> crop {0.25f, 0, 0.5f, 0.75f};
  EXPECT_EQ(preview::detail::aligned_depth_pixel(0, 0, 4, 3, 8, 4, crop, {0, 0}), (std::array<unsigned, 2> {2, 0}));
  EXPECT_EQ(preview::detail::aligned_depth_pixel(3, 2, 4, 3, 8, 4, crop, {0, 0}), (std::array<unsigned, 2> {5, 2}));
  EXPECT_EQ(preview::detail::aligned_depth_pixel(0, 0, 4, 3, 8, 4, crop, {0.125f, 0.25f}), (std::array<unsigned, 2> {3, 1}));
  EXPECT_EQ(preview::detail::aligned_depth_pixel(3, 2, 4, 3, 8, 4, crop, {0.125f, 0.25f}), (std::array<unsigned, 2> {5, 2}));
}

TEST(Game3dDebugPreview, FilenamesRejectTraversalAndAlternateStreams) {
  EXPECT_TRUE(preview::detail::safe_artifact_filename("0_source_color.bin"));
  EXPECT_TRUE(preview::detail::safe_artifact_filename("0_source_color.BIN"));
  for (const auto *bad : {"", ".", "..", "../frame.bin", "..\\frame.bin", "C:frame.bin", "frame.bin:stream", "/frame.bin", "color.png", "COLOR.PNG", "index.html", "visualizations.json", "manifest.preview.tmp", "manifest.json"}) {
    EXPECT_FALSE(preview::detail::safe_artifact_filename(bad));
  }
}

TEST(Game3dDebugPreview, WritesDecodableFullSizePngAndEscapedHtml) {
  preview_directory output;
  auto color = bytes<std::uint8_t>({8, 127, 244, 255, 1, 2, 3, 255});
  auto depth = bytes<float>({0.5f, NAN});
  const std::array<preview::image_view, 2> inputs {{{"source_color", 2, 1, 28, color}, {"raw_depth", 2, 1, 41, depth}}};
  json manifest = {{"status", "complete"}, {"reason", "<script>unsafe</script>"}, {"producer_metadata", {{"color_space", 1}, {"consumed_depth", {{"orientation", 2}}}}}};
  const auto report = preview::generate(output.path, manifest, inputs);
  EXPECT_EQ(report.at("images").size(), 3u);
  EXPECT_TRUE(std::filesystem::exists(output.path / "color.png"));
  EXPECT_TRUE(std::filesystem::exists(output.path / "raw_depth.png"));
  const auto initialized = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
  {
    Microsoft::WRL::ComPtr<IWICImagingFactory> factory;
    ASSERT_TRUE(SUCCEEDED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory))));
    Microsoft::WRL::ComPtr<IWICBitmapDecoder> decoder;
    ASSERT_TRUE(SUCCEEDED(factory->CreateDecoderFromFilename((output.path / "color.png").c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnLoad, &decoder)));
    Microsoft::WRL::ComPtr<IWICBitmapFrameDecode> frame;
    ASSERT_TRUE(SUCCEEDED(decoder->GetFrame(0, &frame)));
    unsigned width = 0, height = 0;
    ASSERT_TRUE(SUCCEEDED(frame->GetSize(&width, &height)));
    EXPECT_EQ(width, 2u);
    EXPECT_EQ(height, 1u);
    Microsoft::WRL::ComPtr<IWICFormatConverter> converter;
    ASSERT_TRUE(SUCCEEDED(factory->CreateFormatConverter(&converter)));
    ASSERT_TRUE(SUCCEEDED(converter->Initialize(frame.Get(), GUID_WICPixelFormat24bppRGB, WICBitmapDitherTypeNone, nullptr, 0, WICBitmapPaletteTypeCustom)));
    std::array<std::uint8_t, 6> decoded {};
    ASSERT_TRUE(SUCCEEDED(converter->CopyPixels(nullptr, 6, 6, decoded.data())));
    EXPECT_EQ(decoded, (std::array<std::uint8_t, 6> {8, 127, 244, 1, 2, 3}));
  }
  if (SUCCEEDED(initialized)) {
    CoUninitialize();
  }
  std::ifstream html(output.path / "index.html");
  const std::string text((std::istreambuf_iterator<char>(html)), {});
  EXPECT_EQ(text.find("<script>unsafe"), std::string::npos);
  EXPECT_NE(text.find("&lt;script&gt;unsafe"), std::string::npos);
}

TEST(Game3dDebugPreview, EmptyDumpStillExplainsMissingImagesAndRejectsMalformedBytes) {
  preview_directory output;
  const auto report = preview::generate(output.path, json::object(), {});
  EXPECT_EQ(report.at("status"), "unavailable");
  EXPECT_TRUE(std::filesystem::exists(output.path / "index.html"));
  auto color = bytes<std::uint8_t>({1, 2, 3});
  const std::array<preview::image_view, 1> inputs {{{"source_color", 1, 1, 28, color}}};
  EXPECT_THROW(preview::generate(output.path, json::object(), inputs), std::runtime_error);
}

TEST(Game3dDebugPreview, ConsumedUiAlphaIsDistinctFromSourceAndLatestOptionalBackbuffer) {
  preview_directory output;
  const auto source = bytes<std::uint8_t>({4, 8, 16, 255, 4, 8, 16, 255});
  const auto ui = bytes<std::uint8_t>({64, 128, 255, 0, 64, 128, 255, 128});
  const auto latest = bytes<std::uint8_t>({0, 0, 0, 255, 0, 0, 0, 255});
  const std::array<preview::image_view, 3> inputs {{{"source_color", 2, 1, 28, source},
    {"ui_source_color", 2, 1, 28, ui}, {"sl_backbuffer", 2, 1, 28, latest}}};
  const json provenance {{"sequence", 42}, {"same_game_frame", "unverified"}};
  const auto report = preview::generate(output.path, {{"producer_metadata", {{"color_space", 2}, {"ui_source", provenance}}}}, inputs);
  EXPECT_EQ(read_png(output.path / "source_alpha.png"), (std::vector<std::uint8_t>{255, 255, 255, 255, 255, 255}));
  EXPECT_EQ(read_png(output.path / "ui_source_alpha.png"), (std::vector<std::uint8_t>{0, 0, 0, 128, 128, 128}));
  EXPECT_EQ(read_png(output.path / "ui_source_color.png"), (std::vector<std::uint8_t>{64, 128, 255, 64, 128, 255}));
  const auto entry = std::find_if(report["images"].begin(), report["images"].end(), [](const auto &image) { return image["file"] == "ui_source_color.png"; });
  ASSERT_NE(entry, report["images"].end());
  EXPECT_EQ(entry->at("provenance"), provenance);
  EXPECT_TRUE(report["optional_errors"].empty());
  const auto broken = bytes<std::uint8_t>({0});
  const std::array<preview::image_view, 1> invalid {{{"ui_source_color", 2, 1, 28, broken}}};
  EXPECT_THROW(preview::generate(output.path, json::object(), invalid), std::runtime_error);
}

TEST(Game3dDebugPreview, OptionalMasksHaveFixedScaleAndColorAlphaHasIndependentUnknownTransfer) {
  preview_directory output;
  auto zero = bytes<std::uint8_t>({0, 0});
  auto one = bytes<std::uint8_t>({255, 255});
  auto ui = bytes<float>({.5f, .5f, .5f, .25f, .5f, .5f, .5f, NAN});
  auto broken = bytes<std::uint8_t>({7});
  const std::array<preview::image_view, 4> inputs {{{"sl_no_warp_mask", 2, 1, 61, zero},
    {"sl_ui_alpha", 2, 1, 61, one}, {"sl_ui_color_alpha", 2, 1, 2, ui},
    {"sl_particle_hint", 2, 1, 999, broken}}};
  json metadata = {{"color_space", 2}, {"render_parameters", {{"depth_rect", {0, 0, .1, .1}}}},
    {"ui_resources", {{{"artifact_id", 9}, {"state", "null"}}}},
    {"optional_captures", {{{"artifact_id", 10}, {"transfer_status", "unknown"}, {"active_rect", {1, 0, 1, 1}}}}}};
  const auto report = preview::generate(output.path, {{"producer_metadata", metadata}}, inputs);
  EXPECT_EQ(read_png(output.path / "sl_no_warp_mask.png"), (std::vector<std::uint8_t>{0, 0, 0, 0, 0, 0}));
  EXPECT_EQ(read_png(output.path / "sl_ui_alpha.png"), (std::vector<std::uint8_t>{255, 255, 255, 255, 255, 255}));
  EXPECT_EQ(read_png(output.path / "sl_ui_color_alpha.png"), (std::vector<std::uint8_t>{128, 128, 128, 128, 128, 128}));
  EXPECT_EQ(read_png(output.path / "sl_ui_color_alpha_alpha.png"), (std::vector<std::uint8_t>{64, 64, 64, 255, 0, 255}));
  EXPECT_EQ(report.at("optional_errors").size(), 1u);
  EXPECT_EQ(report.at("ui_resources"), metadata.at("ui_resources"));
  EXPECT_EQ(report.at("optional_captures"), metadata.at("optional_captures"));
  EXPECT_EQ(ui, bytes<float>({.5f, .5f, .5f, .25f, .5f, .5f, .5f, NAN}));

  metadata["optional_captures"][0]["transfer_status"] = "declared";
  metadata["optional_captures"][0]["color_space"] = 2;
  preview::generate(output.path, {{"producer_metadata", metadata}}, inputs);
  EXPECT_EQ(read_png(output.path / "sl_ui_color_alpha.png")[0], 156);
}

TEST(Game3dDebugPreview, OptionalCatalogExceedsLegacyCapacityWithoutInventingAlpha) {
  preview_directory output;
  auto zero = bytes<std::uint8_t>({0});
  std::vector<preview::image_view> inputs;
  for (const auto &entry : sunshine_game3d::ui_resources::catalog)
    inputs.push_back({entry.file_stem, 1, 1, 61, zero});
  ASSERT_GT(inputs.size(), 10u);
  const auto report = preview::generate(output.path, json::object(), inputs);
  EXPECT_EQ(report.at("images").size(), inputs.size());
  EXPECT_TRUE(report.at("optional_errors").empty());
  EXPECT_FALSE(std::filesystem::exists(output.path / "sl_ui_color_alpha_alpha.png"));
  EXPECT_EQ(read_png(output.path / "sl_ui_color_alpha.png"), (std::vector<std::uint8_t>{0, 0, 0}));
  inputs.resize(::game3d_debug::max_textures + 1, inputs.front());
  EXPECT_THROW(preview::generate(output.path, json::object(), inputs), std::runtime_error);
}

TEST(Game3dDebugPreview, StreamlineBackbufferPreservesNativeAlphaWithoutAssumingUiCoverage) {
  preview_directory output;
  auto rgba = bytes<std::uint32_t>({1023u, 1023u | (1u << 30), 1023u | (2u << 30), 1023u | (3u << 30)});
  const auto original = rgba;
  const std::array<preview::image_view, 1> inputs {{{"sl_backbuffer", 4, 1, 24, rgba}}};
  json metadata = {{"color_space", 2}, {"optional_captures", {{{"artifact_id", 32}, {"transfer_status", "unknown"}}}}};
  const auto report = preview::generate(output.path, {{"producer_metadata", metadata}}, inputs);
  EXPECT_EQ(read_png(output.path / "sl_backbuffer_alpha.png"),
            (std::vector<std::uint8_t>{0, 0, 0, 85, 85, 85, 170, 170, 170, 255, 255, 255}));
  EXPECT_EQ(read_png(output.path / "sl_backbuffer.png"),
            (std::vector<std::uint8_t>{255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0}));
  EXPECT_EQ(rgba, original);
  EXPECT_TRUE(report.at("optional_errors").empty());
  EXPECT_STREQ(sunshine_game3d::ui_resources::find(32)->semantic,
               "final_game_color_before_fg_candidate_not_verified_ui_mask");
}

TEST(Game3dDebugPreview, IntegerMasksKeepLiteralValuesBeforeDisplayClipping) {
  preview_directory output;
  auto mask = bytes<std::uint8_t>({0, 1, 255});
  const std::array<preview::image_view, 1> inputs {{{"sl_no_warp_mask", 3, 1, 62, mask}}};
  const auto report = preview::generate(output.path, json::object(), inputs);
  EXPECT_EQ(preview::detail::sample(inputs[0], 2, 0)[0], 255.f);
  EXPECT_EQ(read_png(output.path / "sl_no_warp_mask.png"), (std::vector<std::uint8_t>{0, 0, 0, 255, 255, 255, 255, 255, 255}));
  EXPECT_EQ(report.at("images")[0].at("range").at("maximum"), 255);
}

TEST(Game3dDebugPreview, SourceAlphaUsesRawChannelWithoutTransferOrSbsSubstitution) {
  preview_directory output;
  auto color = bytes<float>({4, 2, -1, 0, 4, 2, -1, .5f, 4, 2, -1, 1, 4, 2, -1, NAN});
  const auto original = color;
  auto sbs = bytes<std::uint8_t>({1, 2, 3, 255, 1, 2, 3, 255, 1, 2, 3, 255, 1, 2, 3, 255});
  const std::array<preview::image_view, 2> inputs {{{"source_color", 4, 1, 2, color}, {"sbs", 4, 1, 28, sbs}}};
  const auto report = preview::generate(output.path, {{"producer_metadata", {{"color_space", 2}}}}, inputs);
  EXPECT_EQ(read_png(output.path / "source_alpha.png"),
    (std::vector<std::uint8_t>{0, 0, 0, 128, 128, 128, 255, 255, 255, 255, 0, 255}));
  EXPECT_EQ(color, original);
  const auto found = std::find_if(report.at("images").begin(), report.at("images").end(),
    [](const json &entry) { return entry.at("file") == "source_alpha.png"; });
  ASSERT_NE(found, report.at("images").end());
  EXPECT_EQ(found->at("source_artifact"), "source_color");
  EXPECT_EQ(found->at("semantic"), "uninterpreted_source_alpha");
  EXPECT_EQ(found->at("range").at("nonfinite_pixels"), 1);
  EXPECT_FALSE(std::filesystem::exists(output.path / "sbs_alpha.png"));
}

TEST(Game3dDebugPreview, SourceAlphaKeepsConstantsAndDoesNotInventMissingComponents) {
  for (const unsigned alpha : {0u, 255u}) {
    preview_directory output;
    auto color = bytes<std::uint8_t>({0, 0, 0, static_cast<std::uint8_t>(alpha)});
    const std::array<preview::image_view, 1> inputs {{{"source_color", 1, 1, 87, color}}};
    preview::generate(output.path, json::object(), inputs);
    EXPECT_EQ(read_png(output.path / "source_alpha.png"), std::vector<std::uint8_t>(3, static_cast<std::uint8_t>(alpha)));
  }
  preview_directory output;
  auto source = bytes<float>({.25f, .75f});
  auto sbs = bytes<std::uint8_t>({0, 0, 0, 255, 0, 0, 0, 255});
  const std::array<preview::image_view, 2> inputs {{{"source_color", 1, 1, 16, source}, {"sbs", 2, 1, 28, sbs}}};
  preview::generate(output.path, json::object(), inputs);
  EXPECT_FALSE(std::filesystem::exists(output.path / "source_alpha.png"));
}
