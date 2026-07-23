/**
 * @file tests/unit/test_video.cpp
 * @brief Test src/video.*.
 */
#include "../tests_common.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iterator>
#include <src/nvenc/nvenc_base.h>
#include <src/nvenc/nvenc_config.h>
#include <src/video.h>
#include <src/video_colorspace.h>
#include <tuple>

#ifdef _WIN32
namespace platf::dxgi {
  int init();
}
#endif

namespace {
  float apply_color_vector(const float (&color_vector)[4], float red, float green, float blue) {
    return color_vector[0] * red + color_vector[1] * green + color_vector[2] * blue + color_vector[3];
  }

  float reference_srgb_code_to_bt709_code(float value) {
    const auto x = std::clamp(value, 0.0f, 1.0f);
    const auto linear = x <= 0.04045f ?
                          x / 12.92f :
                          std::pow((x + 0.055f) / 1.055f, 2.4f);
    return linear < 0.018f ?
             4.5f * linear :
             1.099f * std::pow(linear, 0.45f) - 0.099f;
  }
}  // namespace

#ifdef _WIN32
TEST(DirectxShaderTest, CompilesAllColorShaderVariants) {
  // D3DCompileFromFile does not require a D3D device. This covers BGRA8, FP16 SDR, PQ,
  // planar luma, both chroma sitings, and the HDR cursor shader in one focused check.
  EXPECT_EQ(platf::dxgi::init(), 0);
}
#endif

TEST(DirectxShaderSourceTest, ConvertsEveryChromaTapBeforeAveraging) {
  const std::string shader_dir =
    SUNSHINE_SOURCE_DIR "/src_assets/windows/assets/shaders/directx/";
  struct shader_family_t {
    const char *entrypoint;
    const char *converter;
  };
  constexpr std::array families {
    shader_family_t {"convert_yuv420_packed_uv_type0_ps.hlsl", "convert_base.hlsl"},
    shader_family_t {"convert_yuv420_packed_uv_type0s_ps.hlsl", "convert_base.hlsl"},
    shader_family_t {"convert_yuv420_packed_uv_type0_ps_linear.hlsl", "convert_linear_base.hlsl"},
    shader_family_t {"convert_yuv420_packed_uv_type0s_ps_linear.hlsl", "convert_linear_base.hlsl"},
    shader_family_t {
      "convert_yuv420_packed_uv_type0_ps_perceptual_quantizer.hlsl",
      "convert_perceptual_quantizer_base.hlsl"
    },
    shader_family_t {
      "convert_yuv420_packed_uv_type0s_ps_perceptual_quantizer.hlsl",
      "convert_perceptual_quantizer_base.hlsl"
    },
  };

  for (const auto &family : families) {
    const auto path = shader_dir + family.entrypoint;
    std::ifstream entry_input(path, std::ios::binary);
    ASSERT_TRUE(entry_input.is_open()) << path;
    const std::string entry {
      std::istreambuf_iterator<char> {entry_input},
      std::istreambuf_iterator<char> {}
    };

    const auto converter =
      entry.find(std::string {"#include \"include/"} + family.converter + '"');
    const auto packed_uv =
      entry.find("#include \"include/convert_yuv420_packed_uv_ps_base.hlsl\"");
    ASSERT_NE(converter, std::string::npos) << family.entrypoint;
    ASSERT_NE(packed_uv, std::string::npos) << family.entrypoint;
    EXPECT_LT(converter, packed_uv) << family.entrypoint;
  }

  const auto path = shader_dir + "include/convert_yuv420_packed_uv_ps_base.hlsl";
  std::ifstream input(path, std::ios::binary);
  ASSERT_TRUE(input.is_open()) << path;

  const std::string shader {
    std::istreambuf_iterator<char> {input},
    std::istreambuf_iterator<char> {}
  };
  const auto sample_begin = shader.find("float3 SampleChromaInput");
  const auto main_begin = shader.find("float2 main_ps", sample_begin);
  ASSERT_NE(sample_begin, std::string::npos);
  ASSERT_NE(main_begin, std::string::npos);

  const auto sample_helper = shader.substr(sample_begin, main_begin - sample_begin);
  EXPECT_NE(
    sample_helper.find(
      "return CONVERT_FUNCTION(image.Sample(def_sampler, tex_coord).rgb);"
    ),
    std::string::npos
  );
  EXPECT_EQ(sample_helper.find("#if"), std::string::npos);

  // All 2/4/6-tap layouts in the shared body must go through the converted sampler. A raw
  // texture fetch or conversion in main_ps would reintroduce post-average conversion.
  const auto filter_body = shader.substr(main_begin);
  EXPECT_EQ(filter_body.find("image.Sample"), std::string::npos);
  EXPECT_EQ(filter_body.find("CONVERT_FUNCTION"), std::string::npos);
  EXPECT_EQ(shader.find("CONVERT_CHROMA_PER_TAP"), std::string::npos);
}

TEST(ColorTransferTest, CompositeSrgbToBt709MatchesReferencePipeline) {
  // Exercise every 16-bit input code. This catches both transfer knees and validates the bounded
  // no-pow shader approximation against an explicit sRGB decode plus BT.709 encode. Its maximum
  // error is less than 0.01 of one 8-bit code step.
  for (unsigned code = 0; code <= 65535; ++code) {
    const auto input = static_cast<float>(code) / 65535.0f;
    EXPECT_NEAR(
      video::srgb_code_to_bt709_code(input),
      reference_srgb_code_to_bt709_code(input),
      3.8e-5f
    ) << "code="
      << code;
  }
}

TEST(ColorTransferTest, CompositeSrgbToBt709ClampsToCodeRange) {
  EXPECT_FLOAT_EQ(video::srgb_code_to_bt709_code(-1.0f), 0.0f);
  EXPECT_FLOAT_EQ(video::srgb_code_to_bt709_code(2.0f), 1.0f);
}

TEST(ColorTransferTest, HdrToSdrToneMapRequiresLinearHdrInputAndSdrTarget) {
  EXPECT_TRUE(video::hdr_to_sdr_tonemap_required(false, true, true));
  EXPECT_FALSE(video::hdr_to_sdr_tonemap_required(false, true, false));
  EXPECT_FALSE(video::hdr_to_sdr_tonemap_required(false, false, true));
  EXPECT_FALSE(video::hdr_to_sdr_tonemap_required(true, true, true));
}

TEST(ColorTransferTest, SbsIntermediatePreservesPrecisionDuringDdupFormatDiscovery) {
  EXPECT_TRUE(video::sbs_intermediate_requires_fp16(true, false, false, false));
  EXPECT_TRUE(video::sbs_intermediate_requires_fp16(false, true, true, false));
  EXPECT_TRUE(video::sbs_intermediate_requires_fp16(false, true, false, true));
  EXPECT_FALSE(video::sbs_intermediate_requires_fp16(false, true, false, false));
  EXPECT_FALSE(video::sbs_intermediate_requires_fp16(false, false, true, true));
}

TEST(HdrNegotiationTest, RequiresHttpAndRtspToSelectTheSameDynamicRange) {
  EXPECT_FALSE(video::hdr_stream_negotiation_is_coherent(true, 0));
  EXPECT_TRUE(video::hdr_stream_negotiation_is_coherent(true, 1));
  EXPECT_TRUE(video::hdr_stream_negotiation_is_coherent(false, 0));
  EXPECT_FALSE(video::hdr_stream_negotiation_is_coherent(false, 1));
}

TEST(NvencHdrMetadataTest, MapsHevcMasteringDisplayAndContentLightUnits) {
  SS_HDR_METADATA source {};
  source.displayPrimaries[0] = {34000, 16000};
  source.displayPrimaries[1] = {13250, 34500};
  source.displayPrimaries[2] = {7500, 3000};
  source.whitePoint = {15635, 16450};
  source.maxDisplayLuminance = 1000;
  source.minDisplayLuminance = 500;
  source.maxContentLightLevel = 1200;
  source.maxFrameAverageLightLevel = 400;

  const auto mapped = nvenc::hdr_metadata_from_sunshine(source, 1);
  ASSERT_TRUE(mapped.mastering_display);
  EXPECT_EQ(mapped.mastering_display->r.x, 34000);
  EXPECT_EQ(mapped.mastering_display->g.x, 13250);
  EXPECT_EQ(mapped.mastering_display->b.x, 7500);
  EXPECT_EQ(mapped.mastering_display->whitePoint.y, 16450);
  EXPECT_EQ(mapped.mastering_display->maxLuma, 10000000u);
  EXPECT_EQ(mapped.mastering_display->minLuma, 500u);
  ASSERT_TRUE(mapped.content_light_level);
  EXPECT_EQ(mapped.content_light_level->maxContentLightLevel, 1200);
  EXPECT_EQ(mapped.content_light_level->maxPicAverageLightLevel, 400);
}

TEST(NvencHdrMetadataTest, MapsAv1MasteringDisplayDenominators) {
  SS_HDR_METADATA source {};
  source.displayPrimaries[0] = {34000, 16000};
  source.displayPrimaries[1] = {13250, 34500};
  source.displayPrimaries[2] = {7500, 3000};
  source.whitePoint = {15635, 16450};
  source.maxDisplayLuminance = 1000;
  source.minDisplayLuminance = 500;

  const auto mapped = nvenc::hdr_metadata_from_sunshine(source, 2);
  ASSERT_TRUE(mapped.mastering_display);
  EXPECT_EQ(mapped.mastering_display->r.x, 44564);
  EXPECT_EQ(mapped.mastering_display->g.y, 45220);
  EXPECT_EQ(mapped.mastering_display->b.x, 9830);
  EXPECT_EQ(mapped.mastering_display->whitePoint.x, 20493);
  EXPECT_EQ(mapped.mastering_display->maxLuma, 256000u);
  EXPECT_EQ(mapped.mastering_display->minLuma, 819u);
  EXPECT_FALSE(mapped.content_light_level);
}

TEST(NvencHdrMetadataTest, OmitsMetadataForH264OrMissingSource) {
  SS_HDR_METADATA source {};
  EXPECT_FALSE(nvenc::hdr_metadata_from_sunshine(source, 0).mastering_display);
  EXPECT_FALSE(
    nvenc::hdr_metadata_from_sunshine(std::nullopt, 1).mastering_display
  );
}

TEST(ColorVectorsTest, LimitedRangeUnormUsesTargetBitDepth) {
  for (const auto bit_depth : {8u, 10u}) {
    const auto *vectors = video::color_vectors_from_colorspace(
      {video::colorspace_e::rec709, false, bit_depth},
      true
    );
    ASSERT_NE(vectors, nullptr);

    const auto max_value = static_cast<float>((1u << bit_depth) - 1u);
    const auto scale = static_cast<float>(1u << (bit_depth - 8u));
    EXPECT_NEAR(apply_color_vector(vectors->color_vec_y, 0.0f, 0.0f, 0.0f), 16.0f * scale / max_value, 1e-6f);
    EXPECT_NEAR(apply_color_vector(vectors->color_vec_y, 1.0f, 1.0f, 1.0f), 235.0f * scale / max_value, 1e-6f);
    EXPECT_NEAR(apply_color_vector(vectors->color_vec_u, 0.0f, 0.0f, 0.0f), 128.0f * scale / max_value, 1e-6f);
    EXPECT_NEAR(apply_color_vector(vectors->color_vec_v, 1.0f, 1.0f, 1.0f), 128.0f * scale / max_value, 1e-6f);
  }
}

TEST(ColorVectorsTest, FullRangeUnormUsesTargetBitDepth) {
  for (const auto bit_depth : {8u, 10u}) {
    const auto *vectors = video::color_vectors_from_colorspace(
      {video::colorspace_e::rec709, true, bit_depth},
      true
    );
    ASSERT_NE(vectors, nullptr);

    const auto max_value = static_cast<float>((1u << bit_depth) - 1u);
    const auto neutral_chroma = static_cast<float>(1u << (bit_depth - 1u)) / max_value;
    EXPECT_NEAR(apply_color_vector(vectors->color_vec_y, 0.0f, 0.0f, 0.0f), 0.0f, 1e-6f);
    EXPECT_NEAR(apply_color_vector(vectors->color_vec_y, 1.0f, 1.0f, 1.0f), 1.0f, 1e-6f);
    EXPECT_NEAR(apply_color_vector(vectors->color_vec_u, 0.0f, 0.0f, 0.0f), neutral_chroma, 1e-6f);
    EXPECT_NEAR(apply_color_vector(vectors->color_vec_v, 1.0f, 1.0f, 1.0f), neutral_chroma, 1e-6f);
  }
}

TEST(ColorVectorsTest, UintOutputRoundsBt2020LimitedValues) {
  const auto *vectors = video::color_vectors_from_colorspace(
    {video::colorspace_e::bt2020, false, 10},
    false
  );
  ASSERT_NE(vectors, nullptr);

  EXPECT_EQ(static_cast<unsigned>(apply_color_vector(vectors->color_vec_y, 0.0f, 0.0f, 0.0f)), 64u);
  EXPECT_EQ(static_cast<unsigned>(apply_color_vector(vectors->color_vec_y, 1.0f, 1.0f, 1.0f)), 940u);
  EXPECT_EQ(static_cast<unsigned>(apply_color_vector(vectors->color_vec_u, 0.0f, 0.0f, 0.0f)), 512u);
  EXPECT_EQ(static_cast<unsigned>(apply_color_vector(vectors->color_vec_v, 1.0f, 0.0f, 0.0f)), 960u);
}

TEST(NvencConfigTest, UsesVerifiedStreamingDefaults) {
  nvenc::nvenc_config config;

  EXPECT_EQ(config.vbv_percentage_increase, 100);
  EXPECT_TRUE(config.hevc_unidirectional_b);
}

TEST(NvencConfigTest, GatesHevcUnidirectionalBFrames) {
  nvenc::nvenc_config config;

  config.hevc_unidirectional_b = false;
  EXPECT_FALSE(nvenc::should_enable_hevc_unidirectional_b(config, 1, true));

  config.hevc_unidirectional_b = true;
  EXPECT_FALSE(nvenc::should_enable_hevc_unidirectional_b(config, 0, true));
  EXPECT_FALSE(nvenc::should_enable_hevc_unidirectional_b(config, 2, true));
  EXPECT_FALSE(nvenc::should_enable_hevc_unidirectional_b(config, 1, false));
  EXPECT_TRUE(nvenc::should_enable_hevc_unidirectional_b(config, 1, true));
}

TEST(NvencConfigTest, ForcesSplitEncodingOnlyForWideModernCodecs) {
  EXPECT_FALSE(nvenc::should_force_split_frame_encoding(false, 1, 7680, 2));
  EXPECT_FALSE(nvenc::should_force_split_frame_encoding(true, 0, 7680, 2));
  EXPECT_FALSE(nvenc::should_force_split_frame_encoding(true, 1, 4096, 2));
  EXPECT_FALSE(nvenc::should_force_split_frame_encoding(true, 1, 7680, 1));
  EXPECT_FALSE(nvenc::should_force_split_frame_encoding(true, 3, 7680, 2));
  EXPECT_TRUE(nvenc::should_force_split_frame_encoding(true, 1, 7680, 2));
  EXPECT_TRUE(nvenc::should_force_split_frame_encoding(true, 2, 8192, 3));
}

TEST(CaptureBackendFailoverTest, RepeatedEarlyDdupFailuresLatchWgc) {
  video::capture_backend_failover_t failover;

  failover.note_capture_result(
    platf::capture_backend_e::ddup,
    platf::capture_e::reinit,
    0,
    std::chrono::milliseconds(100)
  );
  EXPECT_EQ(failover.preferred_backend(), platf::capture_backend_e::ddup);

  failover.note_capture_result(
    platf::capture_backend_e::ddup,
    platf::capture_e::error,
    1,
    std::chrono::milliseconds(100)
  );
  EXPECT_EQ(failover.preferred_backend(), platf::capture_backend_e::wgc);
}

TEST(CaptureBackendFailoverTest, StableDdupTenureForgivesOneOffReinit) {
  video::capture_backend_failover_t failover;
  failover.note_capture_result(
    platf::capture_backend_e::ddup,
    platf::capture_e::reinit,
    0,
    std::chrono::milliseconds(100)
  );
  failover.note_capture_result(
    platf::capture_backend_e::ddup,
    platf::capture_e::reinit,
    1,
    std::chrono::seconds(3)
  );
  failover.note_capture_result(
    platf::capture_backend_e::ddup,
    platf::capture_e::error,
    0,
    std::chrono::milliseconds(100)
  );

  EXPECT_EQ(failover.preferred_backend(), platf::capture_backend_e::ddup);
}

TEST(CaptureBackendFailoverTest, WgcSelectionDoesNotOscillate) {
  video::capture_backend_failover_t failover;
  failover.note_backend_opened(platf::capture_backend_e::wgc);
  failover.note_capture_result(
    platf::capture_backend_e::wgc,
    platf::capture_e::error,
    0,
    std::chrono::milliseconds(10)
  );
  failover.note_capture_result(
    platf::capture_backend_e::ddup,
    platf::capture_e::reinit,
    240,
    std::chrono::seconds(3)
  );

  EXPECT_EQ(failover.preferred_backend(), platf::capture_backend_e::wgc);

  failover.reset();
  EXPECT_EQ(failover.preferred_backend(), platf::capture_backend_e::ddup);
}

struct FramerateX100Test: testing::TestWithParam<std::tuple<std::int32_t, video::rational_t>> {};

TEST_P(FramerateX100Test, ConvertsToExpectedRational) {
  const auto &[x100, expected] = GetParam();
  const auto actual = video::framerate_x100_to_rational(x100);
  EXPECT_EQ(actual.num, expected.num);
  EXPECT_EQ(actual.den, expected.den);
}

INSTANTIATE_TEST_SUITE_P(
  FramerateX100Tests,
  FramerateX100Test,
  testing::Values(
    std::make_tuple(2397, video::rational_t {24000, 1001}),
    std::make_tuple(2398, video::rational_t {24000, 1001}),
    std::make_tuple(2500, video::rational_t {25, 1}),
    std::make_tuple(2997, video::rational_t {30000, 1001}),
    std::make_tuple(3000, video::rational_t {30, 1}),
    std::make_tuple(5994, video::rational_t {60000, 1001}),
    std::make_tuple(6000, video::rational_t {60, 1}),
    std::make_tuple(11988, video::rational_t {120000, 1001}),
    std::make_tuple(23976, video::rational_t {240000, 1001}),
    std::make_tuple(9498, video::rational_t {4749, 50})
  )
);

TEST(HostSbsDimensionsTest, KeepsFourKPerEyeForHevcAndAv1) {
  for (const int video_format : {1, 2}) {
    const auto dimensions = video::host_sbs_output_dimensions(
      3840,
      2160,
      video_format,
      8192,
      8192
    );
    EXPECT_EQ(dimensions.width, 7680);
    EXPECT_EQ(dimensions.height, 2160);
  }
}

TEST(HostSbsDimensionsTest, HonorsStricterConfiguredLimit) {
  const auto dimensions = video::host_sbs_output_dimensions(2560, 1440, 2, 3840, 8192);
  EXPECT_EQ(dimensions.width, 3840);
  EXPECT_EQ(dimensions.height, 1080);
}

TEST(HostSbsDimensionsTest, CapsFiveKPerEyeToCurrentNvencLimit) {
  const auto dimensions = video::host_sbs_output_dimensions(5120, 2160, 2, 8192, 8192);
  EXPECT_EQ(dimensions.width, 8192);
  EXPECT_EQ(dimensions.height, 1728);
}

TEST(HostSbsDimensionsTest, HonorsLowerRuntimeCodecCapability) {
  const auto dimensions = video::host_sbs_output_dimensions(3840, 2160, 1, 8192, 4096);
  EXPECT_EQ(dimensions.width, 4096);
  EXPECT_EQ(dimensions.height, 1152);
}

TEST(HostSbsDimensionsTest, UsesMeasuredH264Capability) {
  const auto dimensions = video::host_sbs_output_dimensions(3840, 2160, 0, 8192);
  EXPECT_EQ(dimensions.width, 4096);
  EXPECT_EQ(dimensions.height, 1152);
}

TEST(VideoPacketLifetimeTest, RetainsBroadcastStateUntilPacketIsConsumed) {
  auto channel = std::make_shared<int>(42);
  std::weak_ptr<int> weak_channel = channel;
  video::packet_raw_generic packet {{0x01}, 1, true};
  packet.channel_data = channel;

  channel.reset();
  EXPECT_FALSE(weak_channel.expired());

  packet.channel_data.reset();
  EXPECT_TRUE(weak_channel.expired());
}
